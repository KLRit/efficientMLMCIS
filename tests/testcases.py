import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'src'))

from assets import *
from options import *


class option_on_assets():
    def __init__(self, option, asset_coll, exact_price = 0, label = "label"):
        self.option = option
        self.assets = asset_coll
        self.exact_price = exact_price
        self.label = label
        self.description = label


benchOP_standard1 = asset_collection([asset(90, 0.03, 0.15)])
benchOP_standard2 = asset_collection([asset(100, 0.03, 0.15)])
benchOP_standard3 = asset_collection([asset(110, 0.03, 0.15)])

benchOP_challenge1 = asset_collection([asset(97, 0.1, 0.01)])
benchOP_challenge2 = asset_collection([asset(98, 0.1, 0.01)])
benchOP_challenge3 = asset_collection([asset(99, 0.1, 0.01)])

p1a_benchOP_option = vanilla_option(100, 1, payofftype = "Call")
p1a_benchOP = option_on_assets(p1a_benchOP_option, benchOP_standard1, exact_price = 2.758443856146076)
p1a1_benchOP = option_on_assets(p1a_benchOP_option, benchOP_standard2, exact_price= 7.485087593912603)
p1a2_benchOP = option_on_assets(p1a_benchOP_option, benchOP_standard3, exact_price= 14.702019669720769)

p1ac_benchOP_option = vanilla_option(100, 0.25, payofftype = "Call")
p1a1c_benchOP = option_on_assets(p1ac_benchOP_option, benchOP_challenge1, exact_price = 0.033913177006141)
p1a2c_benchOP = option_on_assets(p1ac_benchOP_option, benchOP_challenge2, exact_price= 0.512978189232598)
p1a3c_benchOP = option_on_assets(p1ac_benchOP_option, benchOP_challenge3, exact_price= 1.469203342553328)

p1c_benchOP_option = barrier_option(100, 125, 1, payofftype = "Call", knockouttype = "Knock-Out")
p1c1_benchOP = option_on_assets(p1c_benchOP_option, benchOP_standard1, exact_price = 1.822512255945242)
p1c2_benchOP = option_on_assets(p1c_benchOP_option, benchOP_standard2, exact_price = 3.294086516281595)
p1c3_benchOP = option_on_assets(p1c_benchOP_option, benchOP_standard3, exact_price = 3.221591131246868)

p1cc_benchOP_option = barrier_option(100, 125, 0.25, payofftype = "Call", knockouttype = "Knock-Out")
p1c1c_benchOP = option_on_assets(p1cc_benchOP_option, benchOP_challenge1, exact_price = 0.033913177006134)
p1c2c_benchOP = option_on_assets(p1cc_benchOP_option, benchOP_challenge2, exact_price = 0.512978189232598)
p1c3c_benchOP = option_on_assets(p1cc_benchOP_option, benchOP_challenge3, exact_price = 1.469203342553328)


p4a1_benchOP_assets_heston = asset_collection_heston([asset_heston(90, 0.03, 0.25, 0.0225, 2, 0.0225, - 0.5)])
p4a2_benchOP_assets_heston = asset_collection_heston([asset_heston(100, 0.03, 0.25, 0.0225, 2, 0.0225, - 0.5)])
p4a3_benchOP_assets_heston = asset_collection_heston([asset_heston(110, 0.03, 0.25, 0.0225, 2, 0.0225, - 0.5)])
p4a_benchOP_option = vanilla_option(100, 1, payofftype = "Call")
p4a_benchOP = option_on_assets(p4a_benchOP_option, p4a1_benchOP_assets_heston, exact_price = 2.302535842814927)
p4a2_benchOP = option_on_assets(p4a_benchOP_option, p4a2_benchOP_assets_heston, exact_price = 7.379832496149447)
p4a3_benchOP = option_on_assets(p4a_benchOP_option, p4a3_benchOP_assets_heston, exact_price = 2.302535842814927)

p6a1_option_spread_assets = asset_collection([asset(100, 0.03, 0.15), asset(90, 0.03, 0.15)], corr_matrix = [[1, 0.5], [0.5, 1]])
p6a2_option_spread_assets = asset_collection([asset(100, 0.03, 0.15), asset(100, 0.03, 0.15)], corr_matrix = [[1, 0.5], [0.5, 1]])
p6a3_option_spread_assets = asset_collection([asset(100, 0.03, 0.15), asset(110, 0.03, 0.15)], corr_matrix = [[1, 0.5], [0.5, 1]])
p6a4_option_spread_assets = asset_collection([asset(90, 0.03, 0.15), asset(100, 0.03, 0.15)], corr_matrix = [[1, 0.5], [0.5, 1]])
p6a5_option_spread_assets = asset_collection([asset(110, 0.03, 0.15), asset(100, 0.03, 0.15)], corr_matrix = [[1, 0.5], [0.5, 1]])



p6_benchOP_option = spread_option(0,1,payofftype = "Call")

p6a1_benchOP = option_on_assets(p6_benchOP_option, p6a1_option_spread_assets, exact_price=12.021727425647768)
p6a2_benchOP = option_on_assets(p6_benchOP_option, p6a2_option_spread_assets, exact_price=5.978528810578943)
p6a3_benchOP = option_on_assets(p6_benchOP_option, p6a3_option_spread_assets, exact_price=2.500244806693065)
p6a4_benchOP = option_on_assets(p6_benchOP_option, p6a4_option_spread_assets, exact_price=2.021727425647768)
p6a5_benchOP = option_on_assets(p6_benchOP_option, p6a5_option_spread_assets, exact_price=12.500244806693061)

