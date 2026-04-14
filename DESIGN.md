# DESIGN.md

## Product Vision

Build the Streamlit app like a retro Pokemon-inspired analysis handheld:

- half Pokedex scan terminal
- half handheld battle scene
- more like a playable game screen than a dashboard
- still polished enough for a portfolio and class demo

The app should feel like an advanced trainer device, not a generic analytics tool.

## Core Mood

- Nostalgic handheld RPG energy
- Chunky pixel surfaces and beveled panels
- Tactical game readability
- Clean analytical information wrapped in playful game framing

Avoid:

- SaaS-style rounded cards
- soft modern gradients as the main visual language
- default Streamlit spacing and tables
- minimal white-dashboard aesthetics

## Color System

Use this palette as the main design language:

- Cartridge Cream: `#FFF7E3`
- Shell Navy: `#10203C`
- Screen Teal: `#99E3D4`
- Battle Red: `#D94A3A`
- Electric Yellow: `#F7D64A`
- Grass Green: `#67B85F`
- Steel Gray: `#72808E`
- Ink: `#1B1B1B`
- Off Black: `#0D1321`

Type badges can use brighter per-type colors, but every other surface should anchor back to the palette above.

## Typography

- Headings and UI labels: `Press Start 2P`, fallback to monospace
- Readouts and data values: `VT323`, fallback to monospace
- Supporting body copy: monospace / terminal-like fallback

If those fonts are unavailable locally, preserve the retro feeling with:

- uppercase labels
- wide letter spacing on headings
- large monospace numerals

## Surface Rules

- All primary panels are rectangular with hard edges
- Use thick borders and visible pixel-like box shadows
- Prefer inset frames inside cards to create a “device screen” feel
- Use subtle scanline and dot-grid overlays across the app background
- Images should feel like sprites shown inside a game interface
- Core screens should look like they live inside a cartridge shell
- Avoid long stacks of same-weight cards; one hero screen should dominate each page

## Layout Rules

- Add a top status shell that makes the app feel like a device
- Sidebar should feel like a `Trainer Console`
- Page switching should look like game menu tabs, not plain radio buttons
- Every page needs one dominant hero area before the detail panels
- Information density should be balanced:
  - keep the key metrics and explanation visible
  - collapse raw tabular feeling into styled cards, meters, and compact logs

## Component Recipes

### Game Window

- use as the primary framed surface inside a page
- should feel like a screen inside a handheld device
- include a kicker, title, and one focused payload

### Reveal Card

- centered status card for scan outcome
- three states only: `MATCH`, `PARTIAL MATCH`, `MISREAD`
- state color must be obvious from a distance

### Nameplate HP

- used for battle combatants
- contains name, dex id, types, and a bar-like `win edge`
- should read like a Pokemon battle info plate, not a stats card

### Page Switcher

- chunky menu tabs
- hard edges
- selected state uses bright accent fill
- unselected state uses darker shell colors

### Pokemon Profile Card

- sprite frame
- dex number
- pokemon name
- type badges
- one compact metadata strip
- for scan mode, pair this with status lights, signal bars, and scan steps

### Probability Meter

- render like RPG HP / energy bars
- use a strong fill color and a darker track
- percentage labels must be large and legible
- on type page, one main signal should dominate and the rest should feel secondary

### Commentary Panel

- style as professor notes / battle log
- inset panel with striped or terminal-like background
- use short, stacked lines instead of plain markdown bullets
- if used as a dialogue box, include a game-like tab and a `NEXT >` cue

### Command Menu

- use four clear actions such as `FIGHT / INFO / HISTORY / SIGNALS`
- should read like a handheld RPG menu, not tabbed analytics content
- selected state should feel like a pressed game command

### Feature Snapshot

- show as stat tiles or matchup chips
- highlight positive and negative swings clearly
- avoid plain dataframe-first presentation

## Page-Specific Direction

### Type Predictor

Treat this page as a Pokedex scan screen with a reveal sequence:

- hero shows sprite, dex id, signal bars, scan lights, and scan steps
- layout should tell a story:
  - specimen detected
  - signal analysis
  - ground truth reveal
- left side emphasizes the “known profile”
- center uses a reveal card to declare `MATCH / PARTIAL MATCH / MISREAD`
- right side emphasizes the “predicted profile”
- probability section should feel like a `Type Stack HUD`, not a plain ranked chart
- include a `WHO DOES IT LOOK LIKE?` panel for visual misdirection
- explanation should feel like `Professor Notes` or a scan log

### Battle Predictor

Treat this page as a true handheld battle scene:

- strong VS header
- mirrored fighter cards with actual nameplates and a bar-like win edge
- center result HUD for winner and battle momentum
- include a dialogue box that reads like a turn-0 encounter prompt
- use a command menu to switch between:
  - `FIGHT`
  - `INFO`
  - `HISTORY`
  - `SIGNALS`
- history shown as a trainer record log
- explanations should read like combat commentary, not notebook notes

## Motion and Effects

- no heavy animation dependency
- allow subtle glow, blink, or scanline feel through CSS only
- hover states should feel tactile, like pressing a game UI element
- scanning effects should be gentle: flicker, signal bars, silhouette hint
- battle effects should be restrained: ground shadow, HUD emphasis, status glow

## Responsiveness

- desktop: hero + two-column detail layout
- tablet: compact two-column layout when possible
- mobile/narrow widths: stack cards vertically and keep key HUD values above fold

## Implementation Guidance

- Use CSS injection and lightweight HTML in Streamlit
- Keep model APIs untouched
- Reuse the existing prediction payloads
- Prefer reusable helper renderers for badges, pixel cards, meters, and logs
- Prefer reusable UI helpers with clear names:
  - `render_game_window`
  - `render_nameplate_hp`
  - `render_dialog_box`
  - `render_command_menu`
  - `render_reveal_state`
- Keep deployment compatibility with Streamlit Community Cloud
