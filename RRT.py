import random
import pygame
import math

class treeNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.children = []
        self.parent = None
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def get_val(self):
        return self.x, self.y    
    
    def firstpt(self):

        x = random.randint(1,800)
        y = random.randint(1,600)

        dist = sqrt((self.x-x)^2, (self.y-y)^2)

        if dist <= STEPSIZE:
            newNode = treeNode(x,y)
            start.add_child(newNode)

        else:
            slope = (self.y-y)/(self.x-x)

            if slope == 0 :
                newx = self.x + STEPSIZE
                newy = self.y

            else:
                newx = self.x + (STEPSIZE/(sqrt(1+slope^2)))
                newy = self.y + ((STEPSIZE*slope)/(sqrt(1+slope^2)))
                newNode = treeNode(newx,newy)
                start.add_child(newNode)

    def recurse(self,x,y):
        current_dist = 0
        if self.children > 0:
            for i,child in enumerate(self.children):
                current_dist = sqrt((self.children[i].x-x)^2, (self.children[i].y-y)^2)
                dist = sqrt((self.children[i+1].x-x)^2, (self.children[i+1].y-y)^2)
                if current_dist > dist:
                    child.recurse(x,y)

                if dist <= STEPSIZE:
                    newNode = treeNode(x,y)
                    child.add_child(newNode)

                else:
                    slope = (self.y-y)/(self.x-x)

                    if slope == 0 :
                        newx = self.x + STEPSIZE
                        newy = self.y

                    else:
                        newx = self.x + (STEPSIZE/(sqrt(1+slope^2)))
                        newy = self.y + ((STEPSIZE*slope)/(sqrt(1+slope^2)))
                        newNode = treeNode(newx,newy)
                        child.add_child(newNode)

                

    def RRT(self):
        start = treeNode(random.randint(1,800),random.randint(1,600))
        start.firstpt()

        for i in range(100):
            x = random.randint(1,800)
            y = random.randint(1,600)

            start.recurse(x,y)

STEPSIZE = 4
pygame.init()

screen = pygame.display.set_mode((800,600))
pygame.display.set_caption("RRT Path Planning")
clock = pygame.time.Clock()

running = True
start = treeNode(100, 50)

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    # Draw a solid blue circle in the center
    
    x,y = start.get_val()
    startpt = (x,y)
    print(x,y)
    
    endpt = (500,450)

    pygame.draw.circle(screen, (0, 0, 255), startpt, 15)

    pygame.draw.lines(screen, (0,0,0), False, [startpt, endpt], 2)

    pygame.draw.circle(screen, (0, 255, 0), endpt, 10)

    # Flip the display
    pygame.display.flip()

    clock.tick(60)

pygame.quit()
quit()            