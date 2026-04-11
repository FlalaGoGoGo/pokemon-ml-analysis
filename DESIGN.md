# DESIGN.md

## Product Vision

Build the Streamlit app like a retro Pokemon-inspired analysis handheld:

- half Pokedex scan terminal
- half classic handheld battle HUD
- clearly game-like, but still polished enough for a portfolio and class demo

The app should feel like an advanced trainer device, not a generic dashboard.

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

## Layout Rules

- Add a top status shell that makes the app feel like a device
- Sidebar should feel like a `Trainer Console`
- Page switching should look like game menu tabs, not plain radio buttons
- Every page needs one dominant hero area before the detail panels
- Information density should be balanced:
  - keep the key metrics and explanation visible
  - collapse raw tabular feeling into styled cards, meters, and compact logs

## Component Recipes

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

### Probability Meter

- render like RPG HP / energy bars
- use a strong fill color and a darker track
- percentage labels must be large and legible

### Commentary Panel

- style as professor notes / battle log
- inset panel with striped or terminal-like background
- use short, stacked lines instead of plain markdown bullets

### Feature Snapshot

- show as stat tiles or matchup chips
- highlight positive and negative swings clearly
- avoid plain dataframe-first presentation

## Page-Specific Direction

### Type Predictor

Treat this page as a Pokedex scan screen:

- hero shows sprite, dex id, and scan status
- left side emphasizes the “known profile”
- right side emphasizes the “predicted profile”
- probability list should feel like type signal strength
- explanation should feel like research notes from a professor

### Battle Predictor

Treat this page as a versus battle console:

- strong VS header
- mirrored fighter cards
- center result HUD for winner and probabilities
- history shown as battle record log
- explanations should read like combat commentary

## Motion and Effects

- no heavy animation dependency
- allow subtle glow, blink, or scanline feel through CSS only
- hover states should feel tactile, like pressing a game UI element

## Responsiveness

- desktop: hero + two-column detail layout
- tablet: compact two-column layout when possible
- mobile/narrow widths: stack cards vertically and keep key HUD values above fold

## Implementation Guidance

- Use CSS injection and lightweight HTML in Streamlit
- Keep model APIs untouched
- Reuse the existing prediction payloads
- Prefer reusable helper renderers for badges, pixel cards, meters, and logs
- Keep deployment compatibility with Streamlit Community Cloud
