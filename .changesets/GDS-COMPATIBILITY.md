---
rlayout-viewer: minor
rlayout-core: minor
rlayout-gds: minor
rlayout-app: minor
---

### Broader GDS compatibility

Added support for more GDSII file variations so existing layouts can be opened without preprocessing:

- Read gzip-compressed GDS directly, including concatenated gzip members.
- Preserve cell-name byte identities when resolving references, keeping non-UTF-8 names distinct.
- Accept additional metadata and multiple XY records in boundaries and paths, avoiding import failures on these file variations.
- Retain unresolved references as empty cells and recover from selected record and geometry issues with warnings, allowing the rest of a layout to load.
- Handle repeated points, single-point paths, and overlapping path outlines so these geometries do not prevent loading or rendering.

Added progressive background rendering for large hierarchical layouts, with rendering progress and error reporting, to keep navigation responsive while the view completes.
