from order_parser import OrderParser

order_parser = OrderParser("order.txt",100)

F = order_parser.gen_F()

print(F)
print(order_parser.summary())
print(sum(order_parser.summary()))