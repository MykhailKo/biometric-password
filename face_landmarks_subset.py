gesture_landmark_indexes = [
  ### LIPS
  # Lower outer.
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
  # Upper outer (excluding corners).
  185, 40, 39, 37, 0, 267, 269, 270, 409,
  # Lower inner.
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
  # Upper inner (excluding corners).
  191, 80, 81, 82, 13, 312, 311, 310, 415,
  # Lower semi-outer.
  76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306,
  # Upper semi-outer (excluding corners).
  184, 74, 73, 72, 11, 302, 303, 304, 408,
  # Lower semi-inner.
  62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292,
  # Upper semi-inner (excluding corners).
  183, 42, 41, 38, 12, 268, 271, 272, 407,
  ### LEFT EYE
  # Lower contour.
  33, 7, 163, 144, 145, 153, 154, 155, 133,
  # upper contour (excluding corners).
  246, 161, 160, 159, 158, 157, 173,
  # Halo x2 lower contour.
  130, 25, 110, 24, 23, 22, 26, 112, 243,
  # Halo x2 upper contour (excluding corners).
  247, 30, 29, 27, 28, 56, 190,
  # Halo x3 lower contour.
  226, 31, 228, 229, 230, 231, 232, 233, 244,
  # Halo x3 upper contour (excluding corners).
  113, 225, 224, 223, 222, 221, 189,
  # Halo x4 upper contour (no lower because of mesh structure) or
  # eyebrow inner contour.
  35, 124, 46, 53, 52, 65,
  # Halo x5 lower contour.
  143, 111, 117, 118, 119, 120, 121, 128, 245,
  # Halo x5 upper contour (excluding corners) or eyebrow outer contour.
  156, 70, 63, 105, 66, 107, 55, 193,
  ### RIGHT EYE
  # Lower contour.
  263, 249, 390, 373, 374, 380, 381, 382, 362,
  # Upper contour (excluding corners).
  466, 388, 387, 386, 385, 384, 398,
  # Halo x2 lower contour.
  359, 255, 339, 254, 253, 252, 256, 341, 463,
  # Halo x2 upper contour (excluding corners).
  467, 260, 259, 257, 258, 286, 414,
  # Halo x3 lower contour.
  446, 261, 448, 449, 450, 451, 452, 453, 464,
  # Halo x3 upper contour (excluding corners).
  342, 445, 444, 443, 442, 441, 413,
  # Halo x4 upper contour (no lower because of mesh structure) or
  # eyebrow inner contour.
  265, 353, 276, 283, 282, 295,
  # Halo x5 lower contour.
  372, 340, 346, 347, 348, 349, 350, 357, 465,
  # Halo x5 upper contour (excluding corners) or eyebrow outer contour.
  383, 300, 293, 334, 296, 336, 285, 417,
#   ### LEFT CHEEK
#   116, 101, 100, 36, 205, 207, 187, 123, 50,
#   ### RIGHT CHEEK
#   345, 330, 329, 266, 425, 427, 411, 352, 280,
#   ### MOUTH OUTER
#   216, 92, 165, 167, 164, 393, 391, 322, 436, 410, 287, 432, 422, 424, 418, 421, 200, 201, 194, 204, 202, 212, 186
]

anomaly_landmark_indexes = [
    61,  # lips left corner
    291, # lips rigt corner
    0,   # lips top
    17,  # lips bottom
    159, # left eye top
    145, # left eye bottom
    386, # right eye top
    374, # right eye bottom
    55,  # left brow inner
    52,  # left brow center
    285, # right brow inner
    282  # right brow center
]