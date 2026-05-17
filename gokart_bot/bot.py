from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import logging
import math
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from .config import load_config
from .discord_views import ClaimView
from .formatting import format_laps, format_leaderboard, format_myrecords, format_record_channel, format_session, format_user_records
from .models import SessionRecord, now_iso
from .ocr import OcrEngine
from .storage import JsonStore


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEBUG_ARTIFACTS = ("table_warped.png", "paddleocr_overlay.png", "grid_overlay.png", "virtual_rows_overlay.png", "parsed.json")


class GokartBot(commands.Bot):
    def __init__(self, store: JsonStore, config) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!gokart ", intents=intents)
        self.store = store
        self.image_dir = config.image_dir
        self.ocr_executor = ThreadPoolExecutor(max_workers=config.ocr_workers, thread_name_prefix="gokart-ocr")
        self.ocr = OcrEngine(
            max_side=config.ocr_max_side,
            debug_ocr=config.debug_ocr,
            debug_dir=config.debug_dir,
            ocr_lang=config.ocr_lang,
            ocr_device=config.ocr_device,
            ocr_enable_mkldnn=config.ocr_enable_mkldnn,
            ocr_cpu_threads=config.ocr_cpu_threads,
        )

    async def close(self) -> None:
        self.ocr_executor.shutdown(wait=False, cancel_futures=True)
        await super().close()

    async def setup_hook(self) -> None:
        for session in self.store.sessions_with_result_messages():
            self.add_view(ClaimView(self.store, session.id), message_id=session.result_message_id)

        self.tree.add_command(me_command)
        self.tree.add_command(profile_command)
        self.tree.add_command(myrecords_command)
        self.tree.add_command(leaderboard_command)
        self.tree.add_command(session_command)
        self.tree.add_command(heat_command)
        self.tree.add_command(laps_command)
        self.tree.add_command(fix_command)
        self.tree.add_command(unclaim_command)
        self.tree.add_command(record_channel_command)
        self.tree.add_command(set_record_channel_command)
        self.tree.add_command(clear_record_channel_command)
        self.tree.add_command(debug_channel_command)
        self.tree.add_command(set_debug_channel_command)
        self.tree.add_command(clear_debug_channel_command)
        self.tree.add_command(ping_command)
        await self.tree.sync()

    async def on_ready(self) -> None:
        assert self.user is not None
        LOGGER.info("Logged in as %s (%s)", self.user, self.user.id)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.attachments:
            return

        record_channel_id = self.store.get_record_channel_id()
        if record_channel_id is None or message.channel.id != record_channel_id:
            return

        image_attachments = [attachment for attachment in message.attachments if _is_image_attachment(attachment)]
        if not image_attachments:
            return

        for attachment in image_attachments:
            await self.process_attachment(message, attachment)

    async def process_attachment(self, message: discord.Message, attachment: discord.Attachment) -> None:
        progress = await message.reply("正在辨識卡丁車成績表，請稍候...")
        session_id = self.store.next_session_id()
        image_path = self.image_dir / f"session-{session_id}{Path(attachment.filename).suffix.lower() or '.jpg'}"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        await attachment.save(image_path)

        debug_channel_id = self.store.get_debug_channel_id()
        try:
            loop = asyncio.get_running_loop()
            parsed = await loop.run_in_executor(
                self.ocr_executor,
                partial(self.ocr.recognize_lap_sheet, image_path, session_id, force_debug=debug_channel_id is not None),
            )
            raw_ocr = parsed.raw_debug_summary or {"mode": "grid"}
        except Exception as exc:
            LOGGER.exception("OCR failed for %s", attachment.filename)
            await self.send_debug_artifacts(session_id, attachment.filename)
            await progress.edit(content=f"辨識失敗：{exc}")
            return

        session = SessionRecord(
            id=session_id,
            channel_id=message.channel.id,
            source_message_id=message.id,
            image_url=attachment.url,
            author_user_id=message.author.id,
            author_name=message.author.display_name,
            created_at=now_iso(),
            date=parsed.date,
            printed_time=parsed.printed_time,
            heat=parsed.heat,
            ocr_confidence=parsed.ocr_confidence,
            warnings=parsed.warnings,
            raw_ocr=raw_ocr,
            karts=parsed.karts,
        )
        self.store.add_session(session)

        view = ClaimView(self.store, session.id)
        await progress.edit(content=_fit_discord_message(format_session(session)), view=view)
        session.result_message_id = progress.id
        self.store.update_session(session)
        self.add_view(ClaimView(self.store, session.id), message_id=progress.id)
        await self.send_debug_artifacts(session_id, attachment.filename)

    async def refresh_session_message(self, session: SessionRecord) -> None:
        if session.result_message_id is None:
            return
        channel = self.get_channel(session.channel_id) or await self.fetch_channel(session.channel_id)
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return
        message = await channel.fetch_message(session.result_message_id)
        await message.edit(content=_fit_discord_message(format_session(session)), view=ClaimView(self.store, session.id))

    async def send_debug_artifacts(self, session_id: str, source_filename: str) -> None:
        debug_channel_id = self.store.get_debug_channel_id()
        if debug_channel_id is None:
            return
        try:
            channel = self.get_channel(debug_channel_id) or await self.fetch_channel(debug_channel_id)
            if not isinstance(channel, (discord.TextChannel, discord.Thread)):
                LOGGER.warning("Configured debug channel %s is not a text channel or thread", debug_channel_id)
                return
            debug_dir = self.ocr.debug_dir_for_session(session_id)
            files = [discord.File(path, filename=path.name) for name in DEBUG_ARTIFACTS if (path := debug_dir / name).exists()]
            if not files:
                await channel.send(f"Session #{session_id} debug artifacts not found for `{source_filename}`.")
                return
            await channel.send(f"Session #{session_id} debug artifacts for `{source_filename}`", files=files)
        except discord.HTTPException:
            LOGGER.exception("Failed to send debug artifacts for session %s", session_id)


@app_commands.command(name="me", description="查詢自己的卡丁車歷史紀錄")
async def me_command(interaction: discord.Interaction) -> None:
    bot = _bot(interaction)
    content = format_user_records(interaction.user.id, bot.store.all_sessions())
    await interaction.response.send_message(_fit_discord_message(content), ephemeral=True)


@app_commands.command(name="profile", description="查詢自己或指定使用者的卡丁車紀錄")
@app_commands.describe(user="不填則查詢自己")
async def profile_command(interaction: discord.Interaction, user: discord.User | None = None) -> None:
    bot = _bot(interaction)
    target = user or interaction.user
    name = target.display_name if user else None
    content = format_user_records(target.id, bot.store.all_sessions(), name)
    await interaction.response.send_message(_fit_discord_message(content), ephemeral=True)


@app_commands.command(name="myrecords", description="查詢自己最近幾場卡丁車紀錄")
@app_commands.describe(limit="顯示最近幾場，預設 5")
async def myrecords_command(interaction: discord.Interaction, limit: app_commands.Range[int, 1, 20] = 5) -> None:
    bot = _bot(interaction)
    content = format_myrecords(interaction.user.id, bot.store.all_sessions(), int(limit))
    await interaction.response.send_message(_fit_discord_message(content), ephemeral=True)


@app_commands.command(name="leaderboard", description="查詢伺服器卡丁車排行榜")
@app_commands.describe(limit="顯示筆數，預設 10")
async def leaderboard_command(interaction: discord.Interaction, limit: app_commands.Range[int, 1, 25] = 10) -> None:
    bot = _bot(interaction)
    content = format_leaderboard(bot.store.all_sessions(), limit=limit)
    await interaction.response.send_message(_fit_discord_message(content))


@app_commands.command(name="session", description="查詢某場成績表辨識結果")
@app_commands.describe(session_id="紀錄編號，例如 1")
async def session_command(interaction: discord.Interaction, session_id: str) -> None:
    bot = _bot(interaction)
    session = bot.store.get_session(session_id)
    if session is None:
        await interaction.response.send_message(f"找不到紀錄 #{session_id}", ephemeral=True)
        return
    await interaction.response.send_message(_fit_discord_message(format_session(session)), ephemeral=True)


@app_commands.command(name="heat", description="查詢某場 Heat 辨識結果")
@app_commands.describe(heat_id="紀錄編號，例如 1")
async def heat_command(interaction: discord.Interaction, heat_id: str) -> None:
    bot = _bot(interaction)
    session = bot.store.get_session(heat_id)
    if session is None:
        await interaction.response.send_message(f"找不到紀錄 #{heat_id}", ephemeral=True)
        return
    await interaction.response.send_message(_fit_discord_message(format_session(session)), ephemeral=True)


@app_commands.command(name="laps", description="查詢某場完整圈速紀錄")
@app_commands.describe(session_id="紀錄編號，例如 1", kart_no="只顯示特定車號。不填則顯示整場")
async def laps_command(
    interaction: discord.Interaction,
    session_id: str,
    kart_no: app_commands.Range[int, 1, 999] | None = None,
) -> None:
    bot = _bot(interaction)
    session = bot.store.get_session(session_id)
    if session is None:
        await interaction.response.send_message(f"找不到紀錄 #{session_id}", ephemeral=True)
        return
    await _send_text_or_file(
        interaction,
        format_laps(session, int(kart_no) if kart_no else None),
        filename=f"gokart-session-{session_id}-laps.txt",
        ephemeral=True,
    )


@app_commands.command(name="fix", description="手動修正或新增某場的車號圈速")
@app_commands.describe(
    session_id="紀錄編號，例如 1",
    kart_no="車號",
    position="欄位位置。用來把車號未知的第 N 欄改成指定車號",
    best_lap="最佳圈速秒數，例如 19.65。不填則由 laps 自動計算",
    laps="完整圈速，用逗號分隔，例如 20.1,19.65,20.0",
)
async def fix_command(
    interaction: discord.Interaction,
    session_id: str,
    kart_no: app_commands.Range[int, 1, 999],
    position: app_commands.Range[int, 1, 25] | None = None,
    best_lap: float | None = None,
    laps: str | None = None,
) -> None:
    bot = _bot(interaction)
    try:
        parsed_laps = _parse_laps_argument(laps) if laps else None
    except ValueError:
        await interaction.response.send_message("laps 格式錯誤，請用逗號分隔秒數，例如 20.1,19.65,20.0", ephemeral=True)
        return
    try:
        session = bot.store.fix_kart(session_id, int(kart_no), best_lap, parsed_laps, int(position) if position else None)
        try:
            await bot.refresh_session_message(session)
        except discord.HTTPException:
            LOGGER.exception("Failed to refresh session message %s", session_id)
    except ValueError as exc:
        await interaction.response.send_message(str(exc), ephemeral=True)
        return
    await interaction.response.send_message(f"已更新紀錄 #{session_id} 的車號 {kart_no}。", ephemeral=True)


@app_commands.command(name="unclaim", description="取消自己在某場紀錄的認領")
@app_commands.describe(session_id="紀錄編號，例如 1")
async def unclaim_command(interaction: discord.Interaction, session_id: str) -> None:
    bot = _bot(interaction)
    try:
        session = bot.store.unclaim_user(session_id, interaction.user.id)
        try:
            await bot.refresh_session_message(session)
        except discord.HTTPException:
            LOGGER.exception("Failed to refresh session message %s", session_id)
    except ValueError as exc:
        await interaction.response.send_message(str(exc), ephemeral=True)
        return
    await interaction.response.send_message(f"已取消你在紀錄 #{session_id} 的認領。", ephemeral=True)


@app_commands.command(name="recordchannel", description="查看目前設定的紀錄圖片頻道")
async def record_channel_command(interaction: discord.Interaction) -> None:
    bot = _bot(interaction)
    await interaction.response.send_message(format_record_channel(bot.store.get_record_channel_id()), ephemeral=True)


@app_commands.command(name="setrecordchannel", description="設定只有這個頻道的圖片會被辨識成卡丁車紀錄")
@app_commands.describe(channel="紀錄圖片頻道。不填則使用目前頻道")
@app_commands.checks.has_permissions(manage_guild=True)
async def set_record_channel_command(interaction: discord.Interaction, channel: discord.TextChannel | None = None) -> None:
    bot = _bot(interaction)
    target = channel or interaction.channel
    if not isinstance(target, discord.TextChannel):
        await interaction.response.send_message("請在文字頻道使用，或指定一個文字頻道。", ephemeral=True)
        return
    bot.store.set_record_channel_id(target.id)
    await interaction.response.send_message(f"已設定紀錄圖片頻道為 {target.mention}。只有這個頻道的圖片會觸發 OCR。", ephemeral=True)


@app_commands.command(name="clearrecordchannel", description="清除紀錄圖片頻道設定，停止自動辨識圖片")
@app_commands.checks.has_permissions(manage_guild=True)
async def clear_record_channel_command(interaction: discord.Interaction) -> None:
    bot = _bot(interaction)
    bot.store.set_record_channel_id(None)
    await interaction.response.send_message("已清除紀錄圖片頻道設定。bot 目前不會自動辨識任何圖片。", ephemeral=True)


@app_commands.command(name="debugchannel", description="查看目前設定的 OCR debug 圖片頻道")
async def debug_channel_command(interaction: discord.Interaction) -> None:
    bot = _bot(interaction)
    channel_id = bot.store.get_debug_channel_id()
    content = f"目前 OCR debug 頻道：<#{channel_id}>" if channel_id else "目前未設定 OCR debug 頻道。"
    await interaction.response.send_message(content, ephemeral=True)


@app_commands.command(name="setdebugchannel", description="設定 OCR debug artifacts 要傳送到哪個頻道")
@app_commands.describe(channel="OCR debug 頻道。不填則使用目前頻道")
@app_commands.checks.has_permissions(manage_guild=True)
async def set_debug_channel_command(interaction: discord.Interaction, channel: discord.TextChannel | None = None) -> None:
    bot = _bot(interaction)
    target = channel or interaction.channel
    if not isinstance(target, discord.TextChannel):
        await interaction.response.send_message("請在文字頻道使用，或指定一個文字頻道。", ephemeral=True)
        return
    bot.store.set_debug_channel_id(target.id)
    await interaction.response.send_message(
        f"已設定 OCR debug 頻道為 {target.mention}。之後每次辨識都會上傳 debug artifacts，無論 .env debug 是否開啟。",
        ephemeral=True,
    )


@app_commands.command(name="cleardebugchannel", description="清除 OCR debug 圖片頻道設定")
@app_commands.checks.has_permissions(manage_guild=True)
async def clear_debug_channel_command(interaction: discord.Interaction) -> None:
    bot = _bot(interaction)
    bot.store.set_debug_channel_id(None)
    await interaction.response.send_message("已清除 OCR debug 頻道設定。", ephemeral=True)


@app_commands.command(name="ping", description="檢查 bot 是否在線")
async def ping_command(interaction: discord.Interaction) -> None:
    await interaction.response.send_message("pong", ephemeral=True)


def _bot(interaction: discord.Interaction) -> GokartBot:
    if not isinstance(interaction.client, GokartBot):
        raise TypeError("Unexpected bot client")
    return interaction.client


def _is_image_attachment(attachment: discord.Attachment) -> bool:
    suffix = Path(attachment.filename).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return True
    return bool(attachment.content_type and attachment.content_type.startswith("image/"))


def _parse_laps_argument(value: str) -> list[float]:
    laps: list[float] = []
    for part in value.replace("，", ",").split(","):
        stripped = part.strip()
        if not stripped:
            continue
        lap = float(stripped)
        if not math.isfinite(lap) or lap <= 0:
            raise ValueError("Lap time must be a positive finite number")
        laps.append(lap)
    return laps


def _fit_discord_message(content: str) -> str:
    if len(content) <= 1900:
        return content
    return content[:1850] + "\n...（內容過長，已截斷。可用 /session 查詢或檢查 JSON 檔）"


async def _send_text_or_file(interaction: discord.Interaction, content: str, filename: str, ephemeral: bool) -> None:
    if len(content) <= 1900:
        await interaction.response.send_message(content, ephemeral=ephemeral)
        return

    buffer = io.BytesIO(content.encode("utf-8"))
    file = discord.File(buffer, filename=filename)
    await interaction.response.send_message("內容較長，已附上完整文字檔。", file=file, ephemeral=ephemeral)


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config()
    store = JsonStore(config.data_path)
    bot = GokartBot(store, config)
    bot.run(config.discord_token)
