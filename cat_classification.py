#!/usr/bin/env python
import numpy as np

data = np.array([
    ['Pointy', 'Round',     'Present', 1],
    ['Floppy', 'Not round', 'Present', 1],
    ['Floppy', 'Round',     'Absent',  0],
    ['Pointy', 'Not round', 'Present', 0],
    ['Pointy', 'Round',     'Present', 1],
    ['Pointy', 'Round',     'Absent',  1],
    ['Floppy', 'Not round', 'Absent',  0],
    ['Pointy', 'Round',     'Absent',  1],
    ['Floppy', 'Round',     'Absent',  0],
    ['Floppy', 'Round',     'Absent',  0],
])

X, y = data[:, :-1], data[:, -1]
y = y.reshape((-1, 1))

breakpoint()

