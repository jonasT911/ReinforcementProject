import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites

class Pacman(Entity):
    def __init__(self,node):
        Entity.__init__(self, node )
        self.name = PACMAN
        self.color = YELLOW
        self.pointing = LEFT
        self.direction = LEFT
        self.alive = True
        self.setBetweenNodes(LEFT)
        self.sprites = PacmanSprites(self)
    
    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True

    def die(self):
        self.alive = False
        self.direction = STOP
    
    def update(self, dt):
        self.sprites.update(dt)	
        self.position += self.directions[self.direction]*self.speed*dt
        direction = self.getValidKey()
        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)

            if self.target is self.node:
                self.direction = STOP
            self.setPosition()
        else: 
            if self.oppositeDirection(direction):
                self.reverseDirection()


    def machineUpdate(self, dt,action):
        self.sprites.update(dt)	
        direction = self.drivingControl(action)
        self.position += self.directions[self.direction]*self.speed*dt
      
        print("Direction = "+str(direction))
        oldPoint=self.pointing
        self.pointing=direction
        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)

            if self.target is self.node:
                self.direction = STOP
                self.pointing=oldPoint
            self.setPosition()
        else: 
            if self.oppositeDirection(direction):
                self.reverseDirection()
    
    def getValidKey(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_UP]:
            return UP
        if key_pressed[K_DOWN]:
            return DOWN
        if key_pressed[K_LEFT]:
            return LEFT
        if key_pressed[K_RIGHT]:
            return RIGHT
        return STOP
        
    def convertMachineToAction(self,machine):
        key_pressed = pygame.key.get_pressed()
        if machine[0]:#Forward
            return UP
        if machine[1]:
            return DOWN
        if machine[2]:
            return LEFT
        if machine[3]:
            return RIGHT
        return STOP
        
    def drivingControl(self,machine):
        if(self.direction != STOP):
            self.pointing=self.direction
        directions=[LEFT,UP,RIGHT,DOWN]
        i=directions.index(self.pointing)
        key_pressed = pygame.key.get_pressed()
        if machine[0]:#Forward
            return self.pointing
            
        if machine[1]:#LEFT
            return  directions[(i-1)%4]
            
        if machine[2]:#RIGHT
           
            return directions[(i+1)%4]

        if machine[3]: #Reverse
         
            return self.pointing*-1
        return STOP
    def eatPellets(self, pelletList):
        for pellet in pelletList:
             if self.collideCheck(pellet) and (not pellet.eaten):
                pellet.eaten=True
                return pellet
        return None
        
    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    def collideCheck(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False
        
    def MLGhostDistance(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        
        return dSquared
        
    def nearestPellet(self,pelletList):
        nearest=-1
        if(len(pelletList)==0):
            return None,0
        returnPellet=pelletList[0]
       
        for pellet in pelletList:
             if (not pellet.eaten):
                 d = self.position - pellet.position
                 dSquared = d.magnitudeSquared()
                 if(nearest==-1 or dSquared<nearest):
                     nearest=dSquared
                     returnPellet = pellet
                 
        return returnPellet, nearest

