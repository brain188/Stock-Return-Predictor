# Bundled fonts

Served from this folder so the dashboard needs no font service at runtime.
Each file is the Latin subset Google Fonts serves, downloaded once.

| File | Family | Licence |
| --- | --- | --- |
| `Inter-latin.woff2` | Inter (variable, weights 400 to 600) | SIL Open Font License 1.1 |
| `JetBrainsMono-latin.woff2` | JetBrains Mono (variable, weights 400 to 600) | SIL Open Font License 1.1 |
| `MaterialSymbolsOutlined-subset.woff2` | Material Symbols Outlined, only the icons the dashboard uses | Apache License 2.0 |

To add an icon, download the subset again with the new name added to
`icon_names` (see `dashboard/theme.py`, `ICONS`).
