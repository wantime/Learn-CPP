//
// Created by wangyao on 2022/12/29.
//

#ifndef INC_01_TETRIS_TETRIS_H
#define INC_01_TETRIS_TETRIS_H

#include <ctime>
#include <cstdlib>
#include <vector>
#include <graphics.h>
#include "Block.h"

using namespace std;
class Tetris {
public:
    //构造函数
    Tetris(int rows, int cols, int left, int top, int blockSize);
    //初始化
    void init();
    //开始游戏
    void play();
private:
    // 接收用户的输入
    void keyEvent();
    void updateWindow();
    //返回距离上一次调用该函数，间隔了多少时间(ms)
    //第一次调用该函数，返回0
    int getDelay();
    void drop();
    void clearLine();
    void moveLeftRight(int offset);
    void rotate();
    void drawScore();
    void checkOver();
    void saveScore();
    void displayOver();
private:
    int delay;
    // 是否更新
    bool update;

    // int map[20][10]
    // 0：空白，没有任何方块
    // 1-7 是第i种俄罗斯方块
    vector<vector<int>> map;
    int rows;
    int cols;
    int leftMargin;
    int topMargin;
    int blockSize;
    IMAGE imgBg;

    Block* curBlock;
    Block* nextBlock; //预告方块
    Block bakBlock; //当前方块降落过程中用来备份上一个合法位置的！

    int score; //分数
    int level; //当前关卡
    int lineCount; //消除的行数
    int highestScore;
    bool gameOver; // 游戏是否已经结束
};


#endif //INC_01_TETRIS_TETRIS_H
