from order_gen import OrderGen

order_generator = OrderGen(100,30)

o = order_generator.generate_exact(800)
order_generator.save("order.txt")