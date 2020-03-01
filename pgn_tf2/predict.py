def yield_demo():

    for x in range(3):
        for _ in range(3):
            yield x

    print("生成器后一行代码")


a = yield_demo()

print(a)  # 这里的a是一个生成器对象


for i in a:
    print(i)

