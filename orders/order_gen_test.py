from order_gen import OrderGen

for i in range(10):
    order_generator = OrderGen(5,20)
    o = order_generator.generate_exact(64)
    order_generator.save("order"+str(i)+".txt")