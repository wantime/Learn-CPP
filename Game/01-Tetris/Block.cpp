//
// Created by wangyao on 2022/12/29.
//
#include "Block.h"
#include <graphics.h>
#include <time.h>

IMAGE *Block::imgs[7] = {NULL,};
int Block::size = 10;

Block::Block() {

    if (imgs[0] == NULL) {
//        IMAGE imgTmp;
//        loadimage(&imgTmp, "E:\\code\\cpp\\Learn-CPP\\01-Tetris\\res\\tiles-1.png");
//        SetWorkingImage(&imgTmp);
        for (int i = 0; i < 7; i++) {
            imgs[i] = new IMAGE;
            //getimage(imgs[i], i*size, 0, size, size);
        }
        loadimage(imgs[0], "E:\\code\\cpp\\Learn-CPP\\01-Tetris\\res\\0.png", size, size);
        loadimage(imgs[1], "E:\\code\\cpp\\Learn-CPP\\01-Tetris\\res\\1.png", size, size);
        loadimage(imgs[2], "E:\\code\\cpp\\Learn-CPP\\01-Tetris\\res\\2.png", size, size);
        loadimage(imgs[3], "E:\\code\\cpp\\Learn-CPP\\01-Tetris\\res\\3.png", size, size);
        loadimage(imgs[4], "E:\\code\\cpp\\Learn-CPP\\01-Tetris\\res\\4.png", size, size);
        loadimage(imgs[5], "E:\\code\\cpp\\Learn-CPP\\01-Tetris\\res\\5.png", size, size);
        loadimage(imgs[6], "E:\\code\\cpp\\Learn-CPP\\01-Tetris\\res\\6.png", size, size);
//        SetWorkingImage(); //恢复工作区
        srand(time(NULL));
    }
    int blocks[7][4] = {
            1, 3, 5, 7, // I
            2, 4, 5, 7, // Z 1
            3, 5, 4, 6, // Z 2
            3, 5, 4, 7, //
            2, 3, 5, 7,
            3, 5, 7, 6,
            2, 3, 4, 5 // 田
    };

    // 随机生成一种俄罗斯方块
    blockType = 1 + rand() % 7; // 1 + 0...6 => 1...7
    img = imgs[blockType - 1];
    // 初始化smallBlocks
    for (int i = 0; i < 4; i++) {
        int value = blocks[blockType - 1][i];
        smallBlocks[i].row = value / 2;
        smallBlocks[i].col = value % 2;
    }
}

void Block::draw(int leftMargin, int topMargin) {

    for (int i = 0; i < 4; i++) {
        int x = leftMargin + smallBlocks[i].col * size;
        int y = topMargin + smallBlocks[i].row * size;
        //rectangle(x, y, x+size, y+size);
        putimage(x, y, img);
    }
}

IMAGE **Block::getImages() {
    return imgs;
}

void Block::drop() {
    for (int i = 0; i < 4; i++) {
        smallBlocks[i].row++;
    }
//    for(auto &block:smallBlocks){
//        block.row++;
//    }
}

//Block &Block::operator=(const Block &other) {
//    return <#initializer#>;
//}
Block &Block::operator=(const Block &other) {

    if (this == &other) {
        return *this;
    }
    this->blockType = other.blockType;
    for (int i = 0; i < 4; i++) {
        this->smallBlocks[i] = other.smallBlocks[i];
    }
    return *this;
}

bool Block::blockInMap(const vector<vector<int>> &map) {

    int rows = map.size();
    int cols = map[0].size();

    for (int i = 0; i < 4; i++) {
        if (smallBlocks[i].col < 0 || smallBlocks[i].col > cols ||
            smallBlocks[i].row < 0 || smallBlocks[i].row > rows ||
            map[smallBlocks[i].row][smallBlocks[i].col])
            return false;
    }
    return true;
}

void Block::solidify(vector<vector<int>> &map) {
    for (int i = 0; i < 4; i++) {
        // 设置标记，“固化”对应位置
        map[smallBlocks[i].row][smallBlocks[i].col] = blockType;
    }
}

void Block::moveLeftRight(int offset) {
    for (int i = 0; i < 4; i++) {
        smallBlocks[i].col += offset;
    }
}

void Block::rotate() {

    Point p = smallBlocks[1];

    for(int i = 0; i < 4; i ++){
        Point tmp = smallBlocks[i];
        smallBlocks[i].col = p.col - tmp.row + p.row;
        smallBlocks[i].row = p.row + tmp.col - p.col;
    }
}

int Block::getBlockType() {
    return blockType;
}


