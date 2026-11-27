"""Inline SVG icons (simple 24px stroke drawings, nothing loaded from the internet)."""

from __future__ import annotations

from markupsafe import Markup

PATHS = {
    "portfolio": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" '
    'height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" '
    'width="7" height="7" rx="1.5"/>',
    "usage": '<circle cx="12" cy="12" r="9"/><path d="M14.5 9.5c-.5-1-1.5-1.5-2.5-1.5-1.4 0-2.5.8'
    '-2.5 2s1.1 1.6 2.5 2 2.5.8 2.5 2-1.1 2-2.5 2c-1 0-2-.5-2.5-1.5M12 6.5V8M12 16v1.5"/>',
    "summary": '<path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z"/>',
    "forecast": '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/>',
    "reviews": '<path d="M12 3l1.8 4.7 4.7 1.8-4.7 1.8L12 16l-1.8-4.7-4.7-1.8 4.7-1.8z"/>'
    '<path d="M19 15l.8 2.2 2.2.8-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
    "profile": '<path d="M4 21V5a1 1 0 0 1 1-1h9a1 1 0 0 1 1 1v16"/><path d="M15 9h4a1 1 0 0 1 '
    '1 1v11"/><path d="M8 8h3M8 12h3M8 16h3M3 21h18"/>',
    "settings": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12"/><circle cx="16" cy="6" r="2"/>'
    '<circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "cost": '<path d="M6 3h12v18l-3-2-3 2-3-2-3 2z"/><path d="M9 8h6M9 12h6M9 16h3"/>',
    "refresh": '<path d="M20 11a8 8 0 0 0-14.3-4.9L4 8"/><path d="M4 3v5h5"/><path d="M4 13a8 8 '
    '0 0 0 14.3 4.9L20 16"/><path d="M20 21v-5h-5"/>',
    "play": '<path d="M7 4.5l12 7.5-12 7.5z"/>',
    "download": '<path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M4 20h16"/>',
    "alert": '<path d="M12 3.5l9.5 16.5h-19z"/><path d="M12 10v4M12 17.2v.01"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.01"/>',
    "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "logo": '<path d="M4 17l5-5 4 3 7-8"/><path d="M15 7h5v5"/>',
    "estimate": '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 7h8M8 11h2M12 11'
    'h2M8 15h2M12 15h2M8 18h2M12 18h4"/>',
    "clients": '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c.6-3.6 3.3-5.5 6.5-5.5s5.9 1.9 '
    '6.5 5.5"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7M18 14.6c2 .7 3.2 2.6 3.5 5.4"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" '
    'cy="12" r="1"/>',
    "flag": '<path d="M5 21V4M5 4h11l-2 4 2 4H5"/>',
    "edit": '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
}


def icon(name: str) -> Markup:
    return Markup(
        f'<svg class="svg-icon" viewBox="0 0 24 24" aria-hidden="true">{PATHS.get(name, "")}</svg>'
    )
