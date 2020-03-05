from order_gen import OrderGen

# 64 items with 15 SKUs. Generate 4 sets
NO = 64
MAX = 20
NO_SKU = 10
start = ord('a')
for i in range(4):
    order_generator = OrderGen(NO_SKU,MAX)
    o = order_generator.generate_exact(NO)
    order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")
    start += 1

# 300 items with 70 SKUs. Generate 2 sets
NO = 300
MAX = 30
NO_SKU = 50
start = ord('a')
for i in range(2):
    order_generator = OrderGen(NO_SKU,MAX)
    o = order_generator.generate_exact(NO)
    order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")
    start += 1

# 8100 items with 1000 SKUs. Generate one set
NO = 8100
MAX = 50
NO_SKU = 500
start = ord('a')
order_generator = OrderGen(NO_SKU,MAX)
o = order_generator.generate_exact(NO)
order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")
