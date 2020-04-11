from .order_parser import OrderParser
import utils.mtx as mt

order_parser = OrderParser("orders/order_270_30_b.txt",30,0)

F = order_parser.gen_F(is_for_items=False)

d = mt.make_dict(F)

print(d)