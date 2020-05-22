import bpy

def write_rows(matrix, f):
    for r in range(3):
        for c in range(4):
            print(matrix[r][c], end=' ', file=f)
    print(file=f)

f = open('body_to_world.txt', 'w')
for i in range(1, 3001):
    bpy.context.scene.frame_set(i)
    frame_to_world = bpy.context.object.matrix_world
    write_rows(frame_to_world, f)

f.close()
