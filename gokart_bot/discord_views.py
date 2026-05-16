from __future__ import annotations

import discord

from .formatting import format_session
from .storage import JsonStore


class ClaimView(discord.ui.View):
    def __init__(self, store: JsonStore, session_id: str) -> None:
        super().__init__(timeout=None)
        self.store = store
        self.session_id = session_id
        session = store.get_session(session_id)
        if session is None:
            return
        claimable = [kart for kart in session.karts if kart.kart_no is not None]
        for index, kart in enumerate(claimable[:25]):
            disabled = kart.claimed_by_user_id is not None
            button = ClaimButton(session_id, kart.kart_no, disabled=disabled, claimed_by_name=kart.claimed_by_name)
            button.row = index // 5
            self.add_item(button)


class ClaimButton(discord.ui.Button[ClaimView]):
    def __init__(self, session_id: str, kart_no: int, disabled: bool = False, claimed_by_name: str | None = None) -> None:
        super().__init__(
            label=f"車號 {kart_no} 已認領" if claimed_by_name else f"車號 {kart_no}",
            style=discord.ButtonStyle.primary if not disabled else discord.ButtonStyle.secondary,
            custom_id=f"gokart:claim:{session_id}:{kart_no}",
            disabled=disabled,
        )
        self.session_id = session_id
        self.kart_no = kart_no

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        display_name = interaction.user.display_name
        try:
            session = self.view.store.claim_kart(self.session_id, self.kart_no, interaction.user.id, display_name)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return

        new_view = ClaimView(self.view.store, self.session_id)
        await interaction.response.edit_message(content=format_session(session), view=new_view)
        kart = session.find_kart(self.kart_no)
        best = f"{kart.best_lap:.2f}s" if kart and kart.best_lap is not None else "未知"
        await interaction.followup.send(
            f"{interaction.user.mention} 已認領紀錄 #{self.session_id} 的車號 {self.kart_no}。\n本次最佳圈速：{best}",
            ephemeral=True,
        )
