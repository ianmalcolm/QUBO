from order_gen import OrderGen

order_generator = OrderGen(10,5)

o = order_generator.generate_exact(24)
order_generator.save("order.txt")