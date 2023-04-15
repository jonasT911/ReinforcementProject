import torch
import random 
import numpy as np
from collections import deque

import sys
sys.path.append("..") #Import parent folder
from vector import Vector2
from constants import *
from run import GameController

from MLfiles.model import Linear_QNet, QTrainer
from MLfiles.helper import plot

from constants import *

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.0001 #Was .001



class Agent:
    
    def __init__(self):
        self.n_games=0
        self.epsilon = 0 #controls randomness      
        self.gamma=0.6 #discount rate must be smaller than 1
        self.memory = deque(maxlen = MAX_MEMORY) #popleft()
        self.model = Linear_QNet(21,2048,3) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma) 
        self.holdRandom = 0
        self.Random_move = [0,0,0]
        self.previousLocation= Vector2(0,0)
        
      
    def penalizeToLastTurn(self,penalty):
        print("Penalizing")
        i=len(self.memory)-1
        openDirections=0
        action=[1,0,0,0]
        reward=10
        turns=3

        while (i>0 and (turns<2 or len(self.memory)-i<6)):
             state, action, reward, next_state,done=self.memory[i]
             reward=reward+penalty
             turns=int(state[2])+int(state[3])+int(state[4])
             self.memory[i]=(state, action, reward, next_state, done)
            
             i=i-1
        print("Penalty length "+str(len(self.memory)-i))
             
    def rewardUntilLastPenalty(self,extraReward):
        print("Rewarding")
        i=len(self.memory)-1
        if(i<0):
            return
        state, action, old_reward, next_state,done=self.memory[i]
        sameAction=state[6:10]
        openDirections=0
        
        while (i>0 and old_reward>=0):
             state, action, old_reward, next_state,done=self.memory[i]
             openDirections=int(state[2])+int(state[3])+int(state[4])+int(state[5])
             if(old_reward>=0):
                 reward=old_reward+extraReward
                 self.memory[i]=(state, action, reward, next_state, done)
             
             i=i-1
        print("ACTION WAS "+str(len(self.memory)-1-i))
             
    def starve(self,extraReward):
        print("Starving")
        i=len(self.memory)-1
        state, action, old_reward, next_state,done=self.memory[i]
        sameAction=action
        openDirections=0
        
        while (i>0 and old_reward<=0 and len(self.memory)-i<30):
             state, action, old_reward, next_state,done=self.memory[i]
             
             if(old_reward<=0):
                 reward=old_reward+extraReward
                 self.memory[i]=(state, action, reward, next_state, done)
             
             i=i-1
             
             
    def evaluateWholeRun(self,extraReward):
        print("Rewarding")
        i=len(self.memory)-2
        state, action, old_reward, next_state,done=self.memory[i]
        done = False
        print("Run rewards are ")
        print(old_reward)
        while (i>=0 and not done):
             state, action, old_reward, next_state,done=self.memory[i]
           
             if(not done):
                 print(old_reward)
                 reward=old_reward+extraReward
                 self.memory[i]=(state, action, reward, next_state, done)          
             i=i-1
        dist=len(self.memory)-i
        
        print("MEMORY = "+str(len(self.memory)))
        print("RUn was length "+str(dist))
             
    def get_state(self, game):
    
      
#State includes pacman location, ghost location,ghost direction, array of uneaten pellets, available turns.
	#Maybe power pellets, ghosts state(VERY IMPORTANT), fruit, wall location/straight path to ghost? (This should be covered by direction
	#For my paper's purpose: ghost behavior.
	
        pacLoc = game.pacman.position
        print("Pac location is ("+str(pacLoc.x)+", "+str(pacLoc.y)+")")
        
        blink=game.ghosts.blinky
        blinkDistX=pacLoc.x-blink.position.x
        blinkDistY=pacLoc.y-blink.position.y
        
        pink=game.ghosts.pinky
        pinkDistX=pacLoc.x- pink.position.x
        pinkDistY=pacLoc.y- pink.position.y
        
        ink=game.ghosts.inky
        clyde=game.ghosts.clyde
        pellets=game.pellets.pelletList
        
        dir_l = game.pacman.pointing == LEFT
        dir_r = game.pacman.pointing == RIGHT
        dir_u = game.pacman.pointing == UP
        dir_d = game.pacman.pointing == DOWN
        moving=not(pacLoc.x==self.previousLocation.x and pacLoc.y==self.previousLocation.y)
        
        openLeft=game.pacman.getNewTarget(game.pacman.drivingControl([0,0,1,0])) is not game.pacman.node
        openRight=game.pacman.getNewTarget(game.pacman.drivingControl([0,1,0,0])) is not game.pacman.node
        
        nearest,nearDistance=game.pacman.nearestPellet(pellets)
        nearestPP, PPDistance=game.pacman.nearestPellet(game.pellets.powerpellets)
        if(nearestPP==None):
            PPX=0
            PPY=0
        else:
            PPX=pacLoc.x- nearestPP.position.x
            PPY=pacLoc.y- nearestPP.position.y
            
        if(blink.mode.current == FREIGHT):
            blinkDistX=0
            blinkDistY=0
        if(pink.mode.current == FREIGHT):
            pinkDistX=0
            pinkDistY=0
            
        pelletX=pacLoc.x- nearest.position.x
        pelletY=pacLoc.y- nearest.position.y
        
        if(dir_l or dir_r):
            temp=blinkDistX
            blinkDistX=blinkDistY
            blinkDistY=temp
            
            temp=pinkDistX
            pinkDistX=pinkDistY
            pinkDistY=temp
            
            temp=pelletX
            pelletX=pelletY
            pelletY=temp
            
            temp=PPX
            PPX=PPY
            PPY=temp
            
        if(dir_l or dir_u):
            pelletX=pelletX*-1
            pelletY=pelletY*-1
            
            pinkDistX=pinkDistX*-1
            pinkDistY=pinkDistY*-1
            
            blinkDistX=blinkDistX*-1
            blinkDistY=blinkDistY*-1
            
            PPX=PPX*-1
            PPY=PPY*-1
        
        state = [
	
        0,
        0,
        moving,
		openLeft,
		openRight,
		dir_u,
		dir_d,	
		dir_l,
		dir_r,
		
		
		
	#	Ghost Positions
		blinkDistX,
		blinkDistY,
		blink.mode.current == FREIGHT,
		blink.mode.current == SPAWN,
		
		pinkDistX,
		pinkDistY,
		pink.mode.current == FREIGHT,
		pink.mode.current == SPAWN,
	#NOTHING CAN BE PUT IN FRONT OF THIS!
	#	
	#	ink.position.x,
	#	ink.position.y,
	#	ink.mode.current == FREIGHT,
	#	ink.mode.current == SPAWN,
		
	#	clyde.position.x,
	#	clyde.position.y,
	#	clyde.mode.current == FREIGHT,
	#	clyde.mode.current == SPAWN,
		#Ghost Direction 
		
	      
            pelletX,
            pelletY,
            PPX,
            PPY,
         
              
        ]
        
        self.previousLocation=pacLoc
        
        #Add the pellets to the list, and show if they are active.
        blinkDir=[0,0,0,0,0]
        blinkDir[2+blink.direction]=1
        #state.extend(blinkDir)
        
        ink_Dir=[0,0,0,0,0]
        ink_Dir[2+ink.direction]=1
        #state.extend(ink_Dir)
        
        pinkDir=[0,0,0,0,0]
        pinkDir[2+pink.direction]=1
        #state.extend(pinkDir)
        
        clydeDir=[0,0,0,0,0]
        clydeDir[2+clyde.direction]=1
        #state.extend(clydeDir)
        
        
        
        pelletLocations=[]
        for i in pellets:
            if (not i.eaten):
               pelletLocations.extend([i.position.x,i.position.y])#Need better boolean. Visible is for special effects
            else:
               pelletLocations.extend([-1,-1])#Need better boolean. Visible is for special effects
        #state.extend(pelletLocations)
      
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))#If this exceeds max mem then pop left
    
    def train_long_memory(self):
	
        #Can be changed later
        print("Long memory")
        if len(self.memory)>BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #returns list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states,dones = zip (*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    
    def get_action(self,state):
        #random moves: tradeoff exploitation/exploration
        if(self.n_games<1):
        
            self.epsilon = 10
        else:
            self.epsilon = 1 
        if (self.epsilon<0):
            self.epsilon=1#Always ensures a bit of randomness
        final_move = [0,0,0,0]
        if random.randint(0,10)<self.epsilon:
            move =random.randint(0,2) #Dropped to 2 while I can not reverse
            final_move[move] = 1
        else:
            state0 =torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            print("Prediction: "+str(prediction))
            move = torch.argmax(prediction).item() #Change the function to return one of four directions.
            if(prediction[move]<0):
                move =random.randint(0,2) #Dropped to 2 while I can not reverse
            final_move[move]=1
        return final_move
        
def train ():
    static=0
    plot_scores=[]
    plot_mean_scores= []
    total_score=0
    record = 0
    agent = Agent()
    game = GameController(True)
    game.startGame()
    game.MachineLearning=True
    game.lives=1
    mean_score=0
    starving =0
    standingStill=0
    previousLocation= Vector2(0,0)
    
    lastPellet,lastPelletDist=game.pacman.nearestPellet(game.pellets.pelletList)

    while True:
        print("       ")
        #if(agent.n_games==2):
        #    agent.trainer.updateLearningRate(.0001)
        #get old state
        state_old = agent.get_state(game)
        print(state_old)
        
        
        
        #get move
        final_move = agent.get_action(state_old)
        
        #perform move and get new state
        reward,done,score = game.play_step(final_move)
        state_new = agent.get_state(game)
        reward *=10
        if(reward<0):
            agent.penalizeToLastTurn(reward)
        
        
        print(final_move)

        #Penalize getting close to ghosts

    
        print(str(game.pacman.position.x)+" , " +str(game.pacman.position.y)+ " vs " +str(previousLocation.x)+" , " +str(previousLocation.y))
        if (game.pacman.position.x==previousLocation.x and game.pacman.position.y==previousLocation.y):
            print("Standing Still")
            reward=reward-standingStill
            standingStill+=1
        else:
            standingStill=0
        previousLocation=Vector2(game.pacman.position.x,game.pacman.position.y)
        
        if(reward>0):           
            #agent.rewardUntilLastPenalty(reward)#Propagates reward until a different action could have been taken.
            starving=0
        else:
            starving+=1
            if(starving>5):
                
                #reward=reward-.0002*starving
                agent.penalizeToLastTurn(-.00002)
        
        
        nearest,nearDistance=game.pacman.nearestPellet(game.pellets.pelletList)
        if(nearest==lastPellet):
            if (lastPelletDist>nearDistance):
                reward=reward
            else:
                reward=reward-0
        lastPelletDist=nearDistance
        lastPellet=nearest    
        print("Reward is "+str(reward))   
        
        if(not game.pause.paused):
            agent.train_short_memory(state_old,final_move,reward,state_new,done)
            agent.remember(state_old,final_move,reward,state_new,done)
        
        #Game ends
        if done or starving>520:
            runScore=int((game.score -mean_score)/10)
            print("run Score is "+str(runScore))
            #agent.evaluateWholeRun(runScore)
            #train long memory, plot result
            
           
            
            agent.n_games+=1
            #Uncomment this later
            agent.train_long_memory()
            
            if score>record :
                record=score
                agent.model.save()
            print('Game',agent.n_games, 'Score',score,'Record:',record)
            #Plot
            
            plot_scores.append(score)
            total_score+=score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
            
            game.restartGame()
            game.lives=1
            

if __name__== '__main__':
	print("Begin ML Pac-Man")
	train()
