from MLfiles import agent
import sys
from run import GameController


if __name__== '__main__':
    print("Begin ML Pac-Man")
    print ("Number of arguments:" + str(len(sys.argv)) + "arguments.")
    print ("Argument List:"+ str(sys.argv))
    if(len(sys.argv)>1):
        if(sys.argv[1] == "m"):
            if(len(sys.argv)>2):
                agent.train(int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]))
            else:
                agent.train()
        elif (sys.argv[1] == "h"):
            game = GameController()
            game.startGame()
            while True:
                game.update()
    else:
        print("Arguments are h or m")
