"""Shared button styles with consistent hover/pressed/disabled states."""

_PALETTE = {
    "neutral": {
        "bg": "#7b8798",
        "hover": "#8b97a8",
        "pressed": "#6d7888",
        "disabled": "#aeb6c2",
    },
    "primary": {
        "bg": "#3d7e9a",
        "hover": "#4a8faa",
        "pressed": "#326a82",
        "disabled": "#6f8f9d",
    },
    "secondary": {
        "bg": "#5c6bc0",
        "hover": "#6a78cc",
        "pressed": "#4d5aac",
        "disabled": "#8f98ca",
    },
    "success": {
        "bg": "#3d7a5a",   # blue-green (colorblind-safe, not pure green)
        "hover": "#4a8a6a",
        "pressed": "#346a4a",
        "disabled": "#7ea98c",
    },
    "danger": {
        "bg": "#b05055",   # warm coral-red (colorblind-safe, not pure red)
        "hover": "#c06065",
        "pressed": "#9a4045",
        "disabled": "#c9948e",
    },
}


def button_style(variant="primary", padding="10px 20px", radius=4):
    c = _PALETTE.get(variant, _PALETTE["primary"])
    return (
        "QPushButton {"
        f"background-color: {c['bg']};"
        " color: white;"
        " font-weight: bold;"
        f" padding: {padding};"
        f" border-radius: {radius}px;"
        " border: 1px solid rgba(255,255,255,0.10);"
        "}"
        "QPushButton:hover {"
        f" background-color: {c['hover']};"
        "}"
        "QPushButton:pressed {"
        f" background-color: {c['pressed']};"
        "}"
        "QPushButton:disabled {"
        f" background-color: {c['disabled']};"
        " color: #f1f5f9;"
        " border: 1px solid rgba(255,255,255,0.08);"
        "}"
    )
