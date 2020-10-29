from Van import Van
from Courier import Courier


all_couriers = list()
all_orders = list()
all_users = list()
all_vans = list()


def main():
    #create a new van and courier
    van1 = Van(100)
    courier1 = Courier('Mike')
    courier1.van = van1
    print(courier1)


if __name__ == '__main__':

    main()