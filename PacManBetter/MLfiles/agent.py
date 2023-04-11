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
MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.n_games=0
        self.epsilon = 0 #controls randomness      
        self.gamma=0.9 #discount rate must be smaller than 1
        self.memory = deque(maxlen = MAX_MEMORY) #popleft()
        self.model = Linear_QNet(266,256,4) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma) 
      
    
    def get_state(self, game):
    
        print("WHAT")
#State includes pacman location, ghost location,ghost direction, array of uneaten pellets, available turns.
	#Maybe power pellets, ghosts state, fruit
	#Based on test: ghost behavior.
	
        pacLoc = game.pacman.position
	
        dir_l = game.pacman.direction == LEFT
        dir_r = game.pacman.direction == RIGHT
        dir_u = game.pacman.direction == UP
        dir_d = game.pacman.direction == DOWN
        
        blink=game.ghosts.blinky
        pink=game.ghosts.pinky
        ink=game.ghosts.inky
        clyde=game.ghosts.clyde
        pellets=game.pellets.pelletList
        

        
        state = [
		pacLoc.x,
		pacLoc.y,
		  # Move direction
		dir_l,
		dir_r,
		dir_u,
		dir_d,
		
		#Ghost Positions
		blink.position.x,
		blink.position.y,
		
		pink.position.x,
		pink.position.y,
		
		ink.position.x,
		ink.position.y,
		
		clyde.position.x,
		clyde.position.y,
		#Ghost Direction 
		#These need to be changed so that left and right are not twice up and down.
		blink.direction,		
		pink.direction,
		ink.direction,		
		clyde.direction,
		
		#Open positions for pac man
		game.pacman.getNewTarget(LEFT) is not game.pacman.node,
		game.pacman.getNewTarget(RIGHT)is not game.pacman.node,
		game.pacman.getNewTarget(UP) is not game.pacman.node,
		game.pacman.getNewTarget(DOWN)is not game.pacman.node,
		
		
		

        ]
        #Add the pellets to the list, and show if they are active.
      
        pelletLocations=[]
        for i in pellets:
               pelletLocations.extend([int(i.eaten)])#Need better boolean. Visible is for special effects
        
        state.extend(pelletLocations)
        print("THERE ARE peles " +str(len(pelletLocations)))
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
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0,0]
        if random.randint(0,200)<self.epsilon:
            move =random.randint(0,3)
            final_move[move] = 1
        else:
            state0 =torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() #Change the function to return one of four directions.
            final_move[move]=1
        return final_move
        
def train ():
    plot_scores=[]
    plot_mean_scores= []
    total_score=0
    record = 0
    agent = Agent()
    game = GameController()
    game.startGame()
    
    print("At while location")
    while True:
        #get old state
        state_old = agent.get_state(game)
        print(state_old)
        #get move
        final_move = agent.get_action(state_old)
        
        #perform move and get new state
        reward,done,score = game.play_step(final_move)#TODO: This needs to be changed to talk with new AI pacman class
        state_new = agent.get_state(game)
        
        #train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)
        
        #remember
        agent.remember(state_old,final_move,reward,state_new,done)
        
        if done:
            #train long memory, plot result
            game.reset()
            agent.n_games+=1
            #Uncomment this later
            #agent.train_long_memory()
            
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
            
            

if __name__== '__main__':
	print("Begin ML Pac-Man")
	train()
