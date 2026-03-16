from typing import Tuple

import numpy as np
from WindGym.utils.generate_layouts import (
    generate_square_grid,
    generate_cirular_farm,
    generate_right_triangle_grid,
    generate_line_dots_multiple_thetas,
    generate_diamond_grid,
    generate_staggered_grid,
)


def get_layout_positions(layout_type: str, wind_turbine) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get turbine positions for a given layout type.

    Args:
        layout_type: Layout identifier string
        wind_turbine: PyWake wind turbine object

    Returns:
        x_pos, y_pos: Arrays of turbine positions
    """
    layouts = {
        "test_layout": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=1, xDist=5, yDist=5),
        "3turb": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=1, xDist=5, yDist=5),
        "square_2x2": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5),
        "BIG": lambda: generate_square_grid(turbine=wind_turbine, nx=4, ny=4, xDist=6, yDist=6),
        "square_4x4": lambda: generate_square_grid(turbine=wind_turbine, nx=4, ny=4, xDist=6, yDist=6),
        "square_5x5": lambda: generate_square_grid(turbine=wind_turbine, nx=5, ny=5, xDist=6, yDist=6),
        "small_triangle": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=3, xDist=5, yDist=5),
        "square_3x3": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=3, xDist=5, yDist=5),
        "circular_6": lambda: generate_cirular_farm(n_list=[1, 5], turbine=wind_turbine, r_dist=5),
        "circular_10": lambda: generate_cirular_farm(n_list=[3, 7], turbine=wind_turbine, r_dist=5),
        "tri1": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='lower_left'),
        "tri2": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='lower_right'),
        "tri3": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='upper_left'),
        "tri4": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='upper_right'),
        "5turb1": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[0, 30], turbine=wind_turbine),
        "5turb2": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[0, -30], turbine=wind_turbine),
        "5turb3": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[-30, 30], turbine=wind_turbine),
        "a": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=1, xDist=5, yDist=5),
        "b": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5),
        "c": lambda: generate_square_grid(turbine=wind_turbine, nx=4, ny=1, xDist=5, yDist=5),
        "d": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='lower_right'),
        "e": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=2, xDist=5, yDist=5),
        "T1": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=1, xDist=5, yDist=5),
        "T2": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5),
        "T3": lambda: generate_square_grid(turbine=wind_turbine, nx=4, ny=1, xDist=5, yDist=5),
        "T4": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5,orientation="lower_right"),
        "T5": lambda: generate_diamond_grid(wind_turbine, n=2, xDist=5, yDist=2.2),
        "T6": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=3, ny=3, xDist=5, yDist=5,orientation="lower_right"),
        "E1": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=2, xDist=5, yDist=5),
        "E2": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=3, xDist=5, yDist=5),
        "E3": lambda: generate_square_grid(turbine=wind_turbine, nx=4, ny=3, xDist=5, yDist=5),
        "E4": lambda: (lambda x, y: (y, x))(*generate_staggered_grid(turbine=wind_turbine, nx=2, ny=3, xDist=5, yDist=5, y_stagger_offset=[0, 2.5])),
        "E5": lambda: (np.array([730.0624, 444.7964, 1180.635, 93.5193, 1377.9328, 7.4321]),
                       np.array([1016.8061, 452.8746, 437.7612, 1031.7807, 1061.6575, 2.6086])),
        # --- Grid training layouts ---
        "g1": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=2, xDist=5, yDist=5),
        "g2": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=3, xDist=5, yDist=5),
        "g3": lambda: (np.array([5.0, 2.5, 7.5, 0.0, 5.0, 10.0]) * wind_turbine.diameter(),
                        np.array([10.0, 5.0, 5.0, 0.0, 0.0, 0.0]) * wind_turbine.diameter()),
        # --- Perturbed training layouts (~1D perturbation from grid) ---
        "p1": lambda: (np.array([0.9, 5.6, 9.2, -0.5, 4.2, 10.8]) * wind_turbine.diameter(),
                        np.array([-0.7, 1.1, -1.0, 5.8, 4.5, 5.5]) * wind_turbine.diameter()),
        "p2": lambda: (np.array([-0.7, 5.8, 1.0, 4.3, -0.4, 5.5]) * wind_turbine.diameter(),
                        np.array([0.9, -0.5, 5.7, 4.6, 10.8, 9.3]) * wind_turbine.diameter()),
        "p3": lambda: (np.array([5.8, 1.7, 8.4, 0.7, 4.3, 10.7]) * wind_turbine.diameter(),
                        np.array([10.7, 5.7, 4.2, 1.0, -0.7, 0.5]) * wind_turbine.diameter()),
        # --- regular training layouts --- found to match the irregular performance
        "r1": lambda: (np.array([0.0, 1248.1, 2496.2, 0.0, 1248.1, 2496.2]),
                       np.array([0.0, 0.0, 0.0, 891.5, 891.5, 891.5])),
        "r2": lambda: (np.array([0.0, 713.2, 1426.4, 0.0, 713.2, 1426.4]),
                       np.array([0.0, 0.0, 0.0, 713.2, 713.2, 713.2])),
        "r3": lambda: (np.array([0.0, 713.2, 0.0, 713.2, 0.0, 713.2]),
                       np.array([0.0, 0.0, 713.2, 713.2, 1426.4, 1426.4])),
        # --- irregular training layouts --- found to match the regular performance
        "ir1": lambda: (np.array([1704.551048268463, 914.3306311063952, 1228.5520434497569, 258.5255410712933, 23.24574142543186, 1781.9532368498412]),
                        np.array([222.17800042267135, 145.406792381138, 900.5017154692762, 797.6217097748043, 21.26233244323475, 1047.416107140278])),
        "ir2": lambda: (np.array([749.535392785272, 488.31018842263705, 1392.4616322926706, 111.8851795987488, 1747.4330560273631, 1248.5147109705342]),
                        np.array([990.4950250849097, 64.23999669027903, 576.3010846379485, 885.2083834296628, 1059.4419436654625, 59.40283654123481])),
        "ir3": lambda: (np.array([653.1965502153917, 157.89957952486031, 818.9979507784572, 1518.3458685445662, 1190.252321095436, 40.8895846585575]),
                        np.array([213.2061968554637, 698.7844673954643, 1056.6153770780397, 895.3812235989902, 441.623693251274, 159.96676745124617])),
        # --- Evaluation layouts ---
        "eval_grid": lambda: (np.array([0, 5, 10, 15, 2.5, 7.5, 12.5, 17.5]) * wind_turbine.diameter(),
                              np.array([0, 0, 0, 0, 5, 5, 5, 5]) * wind_turbine.diameter()),
        "eval_perturb": lambda: (np.array([1.0, 7.5, 4.0, 11.5, 9.0, 0.5, 14.0]) * wind_turbine.diameter(),
                                 np.array([1.5, 0.0, 6.0, 3.5, 8.5, 11.0, 9.5]) * wind_turbine.diameter()),
        # --- Eval layouts but matching the grid performance ---
        "eval_regular": lambda: (np.array([0.0, 891.5, 1783.0, 0.0, 891.5, 1783.0]), np.array([0.0, 0.0, 0.0, 891.5, 891.5, 891.5])),
        "eval_irregular": lambda: (np.array([1235.1614250818666, 614.0772491446635, 32.67493594582077, 590.169988163652, 70.89867195187087, 1603.8511283851528]),
                                   np.array([872.7611457073143, 47.96788035361005, 951.2489425231132, 972.0429395286828, 366.8012544177941, 95.07779748241096])),
        # --- Larger scale layouts for generalization testing ---
        "20_turb_random_1": lambda: (np.array([1368.8669, 385.5549, 833.9932, 2213.6906, 1469.8883, 881.8675, 2571.9522, 1447.5112,
                                 1667.5234, 105.8911, 511.6958, 2153.1192, 2055.3987, 2664.1796, 332.8692, 45.0046,
                                 1804.4568, 2510.6918, 1274.7111, 13.9351]),
                       np.array([2542.0152, 2537.1629, 1132.1866, 1094.4031, 73.7068, 2108.6526, 1938.4507, 740.5455,
                                 2077.239, 1413.712, 218.1125, 2577.3377, 566.1241, 650.4798, 1961.9877, 811.7346,
                                 1525.975, 60.4911, 1710.2677, 6.5215])),
        "20_turb_random_2": lambda: (np.array([229.0687, 2143.0086, 1281.2227, 1964.6266, 1046.3398, 1973.3472, 3.9852,
                                               798.0741, 1260.5177, 81.1604, 2329.8617, 1502.5601, 571.9873, 2581.5114,
                                               451.8786, 2572.3619, 1404.8292, 553.8118, 2541.4367, 2647.9218]),
                                     np.array([633.3497, 1556.9924, 427.2217, 304.0158, 1382.0216, 2557.5368, 2603.5195,
                                               839.7556, 2068.1294, 1890.7781, 736.4888, 1068.8806, 2296.4378, 1893.9332,
                                               1370.9419, 2562.3502, 2615.497, 122.4296, 137.4661, 1318.8573])),
        "25_turb_random_1": lambda: (np.array([2271.4054, 146.1116, 2900.1217, 2163.2632, 1938.5667, 2909.3338, 3078.096, 1068.7726, 3556.0506, 481.7541, 3330.7992, 1396.5134, 2806.7926, 1198.5934, 1065.1504, 711.4721, 1782.6285, 52.4427, 1711.6369, 2190.855, 444.1621, 2090.1148, 3365.5959, 3301.0982, 2534.9976]), np.array([962.0594, 58.9375, 3254.8864, 2601.3847, 3334.4683, 9.7655, 1930.8507, 1507.3026, 3497.6588, 2572.8274, 1275.8977, 3174.7183, 853.5914, 535.8966, 2396.3337, 3359.5754, 1516.3653, 3079.7406, 828.6418, 101.1509, 1028.1875, 1975.8867, 452.2298, 2668.2542, 1641.6271])),
        "25_turb_random_2": lambda: (np.array([1825.1559, 514.0732, 1111.991, 2951.5874, 1959.8511, 1175.8233, 477.9927, 725.5214, 2675.8004, 3429.2696, 1930.015, 2223.3645, 141.1882, 682.2611, 3049.7394, 767.468, 2401.558, 2964.3261, 1413.0495, 2331.6862, 3552.2395, 60.0061, 3027.6928, 1887.2085, 70.7286]), np.array([3389.3535, 3382.8839, 1509.5821, 1459.2041, 98.2758, 2811.5368, 1437.5009, 935.4094, 999.9376, 2584.6009, 987.394, 2769.652, 1884.9493, 290.8166, 3071.3369, 2279.8557, 3277.47, 223.6521, 20.7705, 1537.7546, 867.3063, 1082.3128, 2159.8661, 2180.8933, 291.9056])),
        "20_turb_test": lambda: (np.array([2271.4054, 146.1116, 2900.1217, 2163.2632, 2909.3338, 3078.096, 1068.7726, 3556.0506, 481.7541, 1396.5134, 1198.5934, 711.4721, 1782.6285, 2211.6812, 2190.855, 3167.0299, 444.1621, 43.1536, 3538.0707, 1191.6717]), np.array([721.5446, 44.2032, 2441.1648, 1951.0386, 7.3241, 1448.138, 1130.477, 2623.2441, 1929.6206, 2381.0388, 401.9224, 2519.6815, 1137.274, 2661.3856, 75.8632, 604.0878, 771.1406, 1291.806, 42.3979, 1772.3138])),
        "25_turb_test": lambda: (np.array([3588.3005, 2297.0637, 240.3961, 1820.7693, 217.3375, 2907.9353, 1938.7787, 1749.1438, 3016.3428, 3921.0451, 1013.2723, 905.9995, 3593.7048, 664.3395, 1999.3853, 3565.463, 2730.2981, 4420.399, 4407.848, 12.2368, 2647.9952, 89.6966, 2509.7429, 1165.3624, 4223.4441]), np.array([2881.1169, 1019.1677, 1367.0934, 161.4513, 3563.062, 836.2634, 3473.948, 1758.1201, 216.8225, 228.9887, 3193.1684, 1158.7455, 1128.4681, 2490.8938, 2849.0182, 1808.205, 3355.6095, 2580.6295, 669.2488, 695.9389, 2629.8073, 1985.8477, 1753.7013, 166.5023, 1334.7882])),
    }

    if layout_type not in layouts:
        raise ValueError(f"Unknown layout type: {layout_type}. Available: {list(layouts.keys())}")

    return layouts[layout_type]()
