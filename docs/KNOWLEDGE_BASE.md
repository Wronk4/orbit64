# N64 Emulator Knowledge Base

## Confirmed Facts & Architectural Conventions

### 1. G_MTX_MUL Post-Multiplication Convention in RSP Geometry Pipeline
- In N64 Fast3D, F3DEX, and OpenGL conventions, successive `G_MTX_MUL` commands perform **post-multiplication**:
  $$\mathbf{M}_{\text{stack}} = \mathbf{M}_{\text{stack}} \times \mathbf{M}_{\text{param}}$$
- In row-vector notation ($v_{\text{clip}} = v \times \mathbf{MV} \times \mathbf{P}$), transformations are applied in sequence:
  - Modelview: `modelview_stack.back() = Matrix4x4::multiply(modelview_stack.back(), mat);`
  - Projection: `projection_matrix = Matrix4x4::multiply(projection_matrix, mat);`
- In games such as Mario Kart 64, the projection matrix is loaded with perspective $\mathbf{P}$ (`G_MTX_LOAD`), followed by multiplication with camera/viewing $\mathbf{V}$ (`G_MTX_MUL`).
- Post-multiplication ($\mathbf{P} \times \mathbf{V}$) ensures track and object vertices transform into positive $w$ clip space. Pre-multiplication erroneously flips or negates $w$.

### 2. Screen-Space Winding Sign Convention with Inverted Y
- In the N64 rasterizer screen space, $Y$ increases downwards:
  $$sy = \text{trans}_y - y_{\text{ndc}} \cdot \text{scale}_y$$
- In this top-down coordinate system, the 2D cross product for signed area:
  $$\text{area} = (x_1 - x_0)(y_2 - y_0) - (x_2 - x_0)(y_1 - y_0)$$
  results in:
  - Counter-clockwise (CCW) front-facing triangles: $\text{area} < 0$
  - Clockwise (CW) back-facing triangles: $\text{area} > 0$
- Backface and frontface culling conditions:
  - Backface cull enabled: reject if $\text{area} \ge 0$
  - Frontface cull enabled: reject if $\text{area} \le 0$

### 3. Sutherland-Hodgman 4D Clipping Rules
- Homogeneous clipping must be performed in 4D clip space before perspective division ($1/w$):
  - Near plane: $w \ge 0.1f$, $w + z \ge 0$
  - Far plane: $w - z \ge 0$
  - Horizontal boundaries: $w \pm x \ge 0$
  - Vertical boundaries: $w \pm y \ge 0$
- Polygon clipping against each plane uses edge interpolation parameter:
  $$t = \frac{d_{\text{prev}}}{d_{\text{prev}} - d_{\text{cur}}}$$
  interpolating clip coordinates $(x, y, z, w)$, texture coordinates $(u, v)$, and color $(r, g, b, a)$.
- Screen coordinates are computed for the resulting clipped polygon vertices, and the polygon is fan-triangulated ($v_0, v_i, v_{i+1}$) for rasterization.

### 4. Framebuffer & VI Switching
- Framebuffer target configured via `G_SETCOLORIMAGE` (0xFF):
  - Bit depth / pixel size: 16-bit RGBA 5-5-5-1 (`color_image_size == 2`) or 32-bit RGBA 8-8-8-8 (`color_image_size == 3`).
  - Line stride in pixels: `color_image_width = (w0 & 0xFFF) + 1`.
  - Framebuffer base in RDRAM: `color_image_addr = segment_to_physical(w1)`.
- VI (Video Interface) scans out framebuffer at `VI_ORIGIN` with line stride `VI_WIDTH`.

### 5. VR4300 TLB Page Sizes and Dual-Page Odd Selection
- In MIPS VR4300, each TLB entry maps an even/odd pair of physical pages.
- Page size per page:
  $$\text{page\_size} = \left(\frac{\text{PageMask}}{2^{13}} + 1\right) \times 4096$$
- The total virtual address range mapped by a single TLB entry is $2 \times \text{page\_size}$ ($\text{mask} = 2 \times \text{page\_size} - 1$).
- The bit distinguishing between the even page (`EntryLo0`) and odd page (`EntryLo1`) is $\text{page\_size}$ (e.g. bit 12 for 4KB, bit 16 for 64KB):
  $$\text{is\_odd} = (\text{vaddr} \ \& \ \text{page\_size}) \ne 0$$
- Testing $(\text{mask} + 1)$ tests the bit above the TLB entry rather than the bit between even and odd pages, which broke 64KB page translations in Super Mario 64 (Segment 4 Goddard animation data).

