# Quiz Results

## Session — 2026-06-13

| # | Topic | Question | Answer | Score |
|---|-------|----------|--------|-------|
| Q1 | Gaussian Blur | Which sigma gives more edge detections — σ=0.5 or σ=2.0, and why? | Smaller sigma keeps image sharper → more edges. Missed: smaller sigma also preserves noise → false positives. | 1/2 |
| Q2 | Canny — NMS | What does non-maximum suppression do and why is it needed? | "New gradients can be computed" — incorrect. NMS thins gradient ridges to 1-pixel-wide edges by keeping only local maxima along the gradient direction. | 0/1 |
| Q3 | Harris | What does R > 0 vs R < 0 indicate, and what is M? | Not attempted. | 0/1 |

**Total: 1 / 4**

---

## Topics to review

- **Canny**: focus on the NMS step — why edges are thick after Sobel, and how NMS picks the ridge peak.
- **Harris**: understand the structure tensor M and what det(M) vs trace(M) tell you about the local neighbourhood.
- **Gaussian blur**: remember the noise trade-off — smaller σ = more edges but also more noise responses.

## Resources

- [Gaussian Blur](gaussian_blur.md)
- [Canny Edge Detection](canny.md)
- [Harris Corner Detection](harris.md)
- [Image Pyramids](pyramids.md)
- [Optical Flow](optical_flow.md)
- [Morphological Operations](morphology.md)
- [Shape from Shading](shape_from_shading.md)
- [Stereo Vision](stereo.md)
