from order_gen import OrderGen

for i in range(10):
    order_generator = OrderGen(15,15)
    o = order_generator.generate_exact(64)
    order_generator.save("order"+str(i)+".txt")