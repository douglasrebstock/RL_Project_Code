import turtle
import math
import time
import random



def square(side):
    for i in range(4):
        bob.forward(side)
        bob.left(90)

def row(n, side):
    for i in range(n):
        square(side)
        bob.forward(side)
    bob.penup()
    bob.left(180)
    bob.forward(n * side)
    bob.left(180)
    bob.pendown()

def row_of_rows(m, n, side):
    for i in range(m):
        row(n, side)
        bob.penup()
        bob.left(90)
        bob.forward(side)
        bob.right(90)
        bob.pendown()
    bob.penup()
    bob.right(90)
    bob.forward(m * side)
    bob.left(90)
    bob.pendown()

original_res = 10
numTilings = 3
numFeatures = math.pow(original_res,2)*numTilings
print(numFeatures)

res_low = 5
res_high = math.sqrt(2*numFeatures/numTilings - math.pow(res_low,2))
featureRange = [math.pow(res_low,2), math.pow(res_high,2)]
print(res_high)
res = []
total_numFeatures = 0
actual_area = 0
for i in range(numTilings):
    area = (featureRange[1] - featureRange[0])/(numTilings-1) * i + featureRange[0]
    total_numFeatures += area
    res.append(math.sqrt(area))  #*(1 + int(math.pow(-1,i))/100))
    actual_area += math.pow(res[-1],2)

print(actual_area)
print(numFeatures)


bob = turtle.Turtle()
bob.speed(0)
origin = (0,0)
offset = [0,0]
colors = ["blue", "red", "green", "brown", "yellow", "pink", "brown", "grey"]
print(res)

for i in range(numTilings):
    bob.color(colors[i])
    #offset[0] += (numTilings + math.pow(-1,i + 1))*res[i]/(numTilings) *3 * math.pow(-1,i + 1)
    #offset[1] += (numTilings + math.pow(-1,i ))*res[i]/(numTilings) *3 * math.pow(-1,i + 1)
    #offset[0] += (numTilings + math.pow(-1,i + 1))*res[i]/(numTilings) *3 * math.pow(-1,i + 1)
    #offset[1] += (numTilings + math.pow(-1,i ))*res[i]/(numTilings) *3 * math.pow(-1,i + 1)
    bob.penup()
    bob.setpos((origin[0] - (15 + 0.5 * math.pow(-1,i + 1)) / res[i] * 200, origin[0] - (15 + 0.5 * math.pow(-1,i)) / res[i] * 200))
    bob.pendown()
    row_of_rows(15,15, 1/res[i] * 400)
bob.penup()
bob.setpos(-200,-200)
bob.pendown()
bob.pensize(5)
bob.color("black")
square(400)
bob.penup()
bob.setpos(-500,-500)
time.sleep(10)



#bob.getscreen().getcanvas().postscript(file="duck.eps")