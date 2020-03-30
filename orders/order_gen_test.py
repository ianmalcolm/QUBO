from order_gen import OrderGen

# 64 items with 15 SKUs. Generate 4 sets
def gen_64():
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
def gen_300():
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
def gen_8100():
    NO = 8100
    MAX = 50
    NO_SKU = 500
    start = ord('a')
    order_generator = OrderGen(NO_SKU,MAX)
    o = order_generator.generate_exact(NO)
    order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")

def gen_3600():
    NO = 3600
    MAX = 50
    NO_SKU = 300
    start = ord('a')
    order_generator = OrderGen(NO_SKU,MAX)
    o = order_generator.generate_exact(NO)
    order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")

def gen_324():
    NO = 324
    MAX = 30
    NO_SKU = 50
    start = ord('a')
    order_generator = OrderGen(NO_SKU,MAX)
    o = order_generator.generate_exact(NO)
    order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")

def gen_169():
    NO = 169
    MAX = 25
    NO_SKU = 40
    start = ord('a')
    order_generator = OrderGen(NO_SKU,MAX)
    o = order_generator.generate_exact(NO)
    order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")

def gen_196():
    NO = 196
    MAX = 25
    NO_SKU = 40
    start = ord('a')
    order_generator = OrderGen(NO_SKU,MAX)
    o = order_generator.generate_exact(NO)
    order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")

def gen_144():
    NO = 144
    MAX = 20
    NO_SKU = 30
    start = ord('a')
    order_generator = OrderGen(NO_SKU,MAX)
    o = order_generator.generate_exact(NO)
    order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")

def gen_8():
    NO = 8
    MAX = 3
    NO_SKU = 3
    start = ord('a')
    order_generator = OrderGen(NO_SKU,MAX)
    o = order_generator.generate_exact(NO)
    order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")


def gen_90():
    NO = 90
    MAX = 10
    NO_SKU = 10
    start = ord('a')
    order_generator = OrderGen(NO_SKU,MAX)
    o = order_generator.generate_exact(NO)
    order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")

def gen_180():
    NO = 180
    MAX = 20
    NO_SKU = 30
    start = ord('a')
    order_generator = OrderGen(NO_SKU,MAX)
    o = order_generator.generate_exact(NO)
    order_generator.save("order"+"_"+str(NO)+"_"+str(NO_SKU)+"_"+chr(start)+".txt")

gen_180()