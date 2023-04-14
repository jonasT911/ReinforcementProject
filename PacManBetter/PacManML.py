from MLfiles import agent
import sys



if __name__== '__main__':
	print("Begin ML Pac-Man")
	print ("Number of arguments:" + str(len(sys.argv)) + "arguments.")
	print ("Argument List:"+ str(sys.argv))
	agent.train()

