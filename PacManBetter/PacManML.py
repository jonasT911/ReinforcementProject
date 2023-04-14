from MLfiles import agent
import sys
from run import GameController


if __name__== '__main__':
	print("Begin ML Pac-Man")
	print ("Number of arguments:" + str(len(sys.argv)) + "arguments.")
	print ("Argument List:"+ str(sys.argv))
	if(sys.argv[1] == "m"):
		agent.train()
	elif (sys.argv[1] == "h"):
		game = GameController()
		game.startGame()
		while True:
			game.update()
	else:
		print("Arguments are h m and e")
