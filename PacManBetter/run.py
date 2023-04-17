import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup 
from pellets import PelletGroup
from ghosts import GhostGroup
from pauser import Pause
from fruit import Fruit
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites

import copy

class GameController(object): #TODO: Add play step function

    MachineLearning=False
    
    def __init__(self,ML=False):
        pygame.init()
        self.MachineLearning=ML
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.clock = pygame.time.Clock()
        self.level = 0
        self.lives = 5
        self.fruit = None
        self.score = 0
        self.textgroup = TextGroup()
        if (not self.MachineLearning):
            self.pause = Pause(True)
        else:
            self.pause = Pause(False)
            self.textgroup.hideText()
       
        self.lifesprites = LifeSprites(self.lives) 

    def restartGame(self,MakePP=True):
        self.lives = 5
        self.level = 0
        
        self.fruit = None
        self.startGame(MakePP)
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        if (not self.MachineLearning):
            self.pause.paused = True
            self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)

    def resetLevel(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def nextLevel(self):
        self.showEntities()
        self.level += 1       
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BLACK)

    def startGame(self,makePowerPellets=True):
        mazeType="maze1.txt"
        self.setBackground()
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.background = self.mazesprites.constructBackground(self.background, self.level%5)
        
        self.nodes = NodeGroup(mazeType)
        self.nodes.setPortalPair((0,17), (27,17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12,14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15,14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup(mazeType,makePowerPellets)
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman )
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        #Deny Access To Home
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
 
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)


    def update(self):
        dt = self.clock.tick(30) / 1000.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.pacman.update(dt)
            self.ghosts.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()
        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()
    

    #TODO: Dont train on paused game
    def play_step(self,action): #This is for machine learning code
        old_score=self.score
        deathPenalty=0
        gameWon=False
        dt = self.clock.tick(30) / 1000.0 #Can I change this?
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.pacman.machineUpdate(dt,action)
            self.ghosts.update(dt)

            if self.fruit is not None:
                self.fruit.update(dt)
            gameWon=self.checkPelletEvents()
            deathPenalty = self.checkGhostEvents()
            self.checkFruitEvents()
        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render() 
        game_over=(self.lives==0)
        game_over=(game_over or gameWon)
        #I may change reward function later
        # I need a penalty for being caught by the ghost
    
        reward=self.score-old_score+deathPenalty
        return reward,game_over,self.score                
                             
    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)                             
                
    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()            
                


    def render(self):
        self.screen.blit(self.background, (0, 0))
     
        self.pellets.render(self.screen)
        self.pacman.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)
        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))
            
        pygame.display.update()
        
    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20))
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
               self.updateScore(self.fruit.points)
               self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
               self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None
            
    def checkGhostEvents(self):
     score=0
     for ghost in self.ghosts:
        if self.pacman.collideGhost(ghost):
            if ghost.mode.current is FREIGHT:
                self.pacman.visible = False
                ghost.visible = False
                #Disabling eating bonus for now
                #self.updateScore(ghost.points)
                self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                self.ghosts.updatePoints()
                self.pause.setPause(pauseTime=1, func=self.showEntities)
                ghost.startSpawn()
                self.nodes.allowHomeAccess(ghost)
            elif ghost.mode.current is not SPAWN:
                     if self.pacman.alive:
                        
                         score=-200
                         self.lives -=  1
                         self.lifesprites.removeImage()
                         self.pacman.die()
                         self.ghosts.hide()
                         if self.lives <= 0:      
                             #Does not reset if I am in ML mode
                             if (self.MachineLearning==False):
                                 self.pause.setPause(pauseTime=3, func=self.restartGame)
                         else:
                             if (self.MachineLearning==False):
                                 self.pause.setPause(pauseTime=3, func=self.resetLevel  )
                             else: 
                                 self.pacman.reset()
                                 self.ghosts.reset()
                                 self.fruit = None      
                     
     return score
      
    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            self.hideEntities()
                            
    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            self.pellets.eatPellet(pellet)#Needs to be changed to not screwUp ML graph
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            if pellet.name == POWERPELLET:
                pass
                self.ghosts.startFreight()
            if self.pellets.finishedLevel():#This will need to be changed after I change pellets
                self.hideEntities()
                print("Victory")
                if (self.MachineLearning==False):
                    self.pause.setPause(pauseTime=3, func=self.nextLevel)
                else: 
                    self.pacman.reset()
                    self.ghosts.reset()
                    self.fruit = None   
                    return True 
        return False


  
if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()
