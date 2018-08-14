-- | ignore

import "linalg"

module linalg_i32 = mk_linalg i32
module linalg_f32 = mk_linalg f32

-- ==
-- entry: test_dotprod
-- input { [0]   [0] }
-- output { 0 }
-- input { [1,2,3] [4,5,6] }
-- output { 32 }

entry test_dotprod [n] (as: [n]i32, bs: [n]i32): i32 =
  linalg_i32.dotprod as bs

-- ==
-- entry: test_outer
-- input { [1,2,3] [5,2,3] }
-- output { [[5, 2, 3],
--           [10, 4, 6],
--           [15, 6, 9]] }

entry test_outer = linalg_i32.outer

-- ==
-- entry: test_inv
-- input { [[1.0f32, 2.0f32, 1.0f32], [2.0f32, 1.0f32, 1.0f32], [1.0f32, 1.0f32, 2.0f32]] }
-- output { [[-0.25f32, 0.75f32, -0.25f32], [0.75f32, -0.25f32, -0.25f32], [-0.25f32, -0.25f32, 0.75f32]] }

entry test_inv [n] (A: [n][n]f32): [n][n]f32 =
  linalg_f32.inv A

-- ==
-- entry: test_matmul
-- input {
--   [ [1,2], [3,4] ]
--   [ [5,6], [7,8] ]
-- }
-- output { [[19, 22], [43, 50]] }

entry test_matmul (x: [][]i32, y: [][]i32): [][]i32 =
  linalg_i32.matmul x y

-- ==
-- entry: test_ols
-- input {
--   [[-1f32, 0f32], [0.98f32, -0.2f32], [-0.32f32, -0.95f32], [-0.71, -0.71]]
--   [-1.77f32, 1.72f32, -2.41f32, -2.81f32]
-- }
-- output { [1.9734432f32, 1.8890195f32] }

entry test_ols [n][m] (X: [n][m]f32) (b: [n]f32): [m]f32 = linalg_f32.ols X b
