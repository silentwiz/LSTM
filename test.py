def cal_ave(some_list):
    a = 0
    for value in some_list:
        a += value
    result = a / len(some_list)
    print(f"{result:.5f}")

class Some_data:
    def __init__(self):
        self.loss = [0.3920358121395111,0.39099884033203125,0.3968547284603119,0.39994633197784424,0.3909059762954712]
        self.acc = [0.01005025114864111,0.027638191357254982,0.027638191357254982,0.02512562833726406,0.032663315534591675]

def main():
    some_data = Some_data()
    cal_ave(some_data.acc)
    
    

if __name__ == "__main__":
    main()