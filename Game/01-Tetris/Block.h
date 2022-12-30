//
// Created by wangyao on 2022/12/29.
//

#ifndef INC_01_TETRIS_BLOCK_H
#define INC_01_TETRIS_BLOCK_H

#include <cstdlib>
#include <graphics.h>
#include <vector>

using namespace std;
struct Point {
    int row;
    int col;
};

class Block {
public:
    Block();// 构造函数
    void drop();// 下降
    void moveLeftRight(int offset);// 左移右移
    void rotate();// 旋转
    void draw(int leftMargin, int topMargin);// 绘制
    static IMAGE **getImages();
    Block& operator=(const Block& other);

    bool blockInMap(const vector<vector<int>> &map);
    //Point * getSmallBlocks();
    void solidify(vector<vector<int>> &map);
    int getBlockType();
private:
    int blockType; //方块的类型
    Point smallBlocks[4]; //方块的行列
    IMAGE *img;

    static IMAGE *imgs[7];
    static int size;
};

#endif //INC_01_TETRIS_BLOCK_H
