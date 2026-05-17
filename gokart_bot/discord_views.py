from __future__ import annotations

import discord

from .formatting import format_kart_laps, format_session
from .storage import JsonStore


class ClaimView(discord.ui.View):
    def __init__(self, store: JsonStore, session_id: str) -> None:
        super().__init__(timeout=None)
        self.store = store
        self.session_id = session_id
        session = store.get_session(session_id)
        if session is None:
            return
        claimable = [kart for kart in session.karts if kart.position is not None]
        for index, kart in enumerate(claimable[:20]):
            disabled = kart.claimed_by_user_id is not None
            button = ClaimButton(session_id, kart.position, kart.kart_no, disabled=disabled, claimed_by_name=kart.claimed_by_name)
            button.row = index // 5
            self.add_item(button)
        if claimable:
            self.add_item(KartLapsSelect(session_id, claimable[:25]))


class ClaimButton(discord.ui.Button[ClaimView]):
    def __init__(self, session_id: str, position: int, kart_no: int | None, disabled: bool = False, claimed_by_name: str | None = None) -> None:
        label = f"車號 {kart_no}" if kart_no is not None else f"欄位 {position}"
        super().__init__(
            label=f"{label} 已認領" if claimed_by_name else label,
            style=discord.ButtonStyle.primary if not disabled else discord.ButtonStyle.secondary,
            custom_id=f"gokart:claim:{session_id}:pos:{position}",
            disabled=disabled,
        )
        self.session_id = session_id
        self.position = position
        self.kart_no = kart_no

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        display_name = interaction.user.display_name
        try:
            session = self.view.store.claim_position(self.session_id, self.position, interaction.user.id, display_name)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return

        new_view = ClaimView(self.view.store, self.session_id)
        await interaction.response.edit_message(content=_fit_discord_message(format_session(session)), view=new_view)
        kart = next((candidate for candidate in session.karts if candidate.position == self.position), None)
        label = f"車號 {kart.kart_no}" if kart and kart.kart_no is not None else f"欄位 {self.position}"
        best = f"{kart.best_lap:.2f}s" if kart and kart.best_lap is not None else "未知"
        await interaction.followup.send(
            f"{interaction.user.mention} 已認領紀錄 #{self.session_id} 的{label}。\n本次最佳圈速：{best}",
            ephemeral=True,
        )


class KartLapsSelect(discord.ui.Select[ClaimView]):
    def __init__(self, session_id: str, karts) -> None:
        options = []
        for kart in karts:
            label = f"車號 {kart.kart_no}" if kart.kart_no is not None else f"欄位 {kart.position}"
            best = f"最佳 {kart.best_lap:.2f}s" if kart.best_lap is not None else "最佳未知"
            laps = f"{len(kart.laps)} 圈" if kart.laps else "圈數未知"
            options.append(
                discord.SelectOption(
                    label=label,
                    value=str(kart.position),
                    description=f"{best}，{laps}",
                )
            )
        super().__init__(
            placeholder="選取車號查看完整圈速",
            min_values=1,
            max_values=1,
            options=options,
            custom_id=f"gokart:laps:{session_id}",
            row=4,
        )
        self.session_id = session_id

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        session = self.view.store.get_session(self.session_id)
        if session is None:
            await interaction.response.send_message(f"找不到紀錄 #{self.session_id}", ephemeral=True)
            return
        position = int(self.values[0])
        kart = next((candidate for candidate in session.karts if candidate.position == position), None)
        if kart is None:
            await interaction.response.send_message(f"找不到欄位 {position}", ephemeral=True)
            return
        await interaction.response.send_message(_fit_discord_message(format_kart_laps(session, kart)), ephemeral=True)


def _fit_discord_message(content: str) -> str:
    if len(content) <= 1900:
        return content
    return content[:1850] + "\n...（內容過長，已截斷。可用 /laps 查詢完整圈速。）"
