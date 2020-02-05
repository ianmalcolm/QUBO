from order_gen import OrderGen

order_generator = OrderGen(15,15)

o = order_generator.generate_exact(64)
order_generator.save("order.txt")