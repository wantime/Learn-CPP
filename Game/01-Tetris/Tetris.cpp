//
// Created by wangyao on 2022/12/29.
//
#include "Tetris.h"
#include "Block.h"
#include <graphics.h>
#include <conio.h>
#include <string>
#include <fstream>
#include <iostream>

#define MAX_LEVEL 5
#define RECORDER_FILE "record.txt"


//const int SPEED_NORMAL = 500; //ms
const int SPEED_NORMAL[MAX_LEVEL] = {500, 300, 150, 100, 80};
const int SPEED_FAST = 30;

void Tetris::play() {
    init();

    nextBlock = new Block;
    curBlock = nextBlock;
    nextBlock = new Block;

    int timer = 0;
    while (1) {
        // 接受用户的输入
        keyEvent();

        timer += getDelay();
        if (timer > delay) {
            timer = 0;
            drop();
            // 渲染游戏画面
            update = true;
        }
        if (update) {
            update = false;
            // 更新游戏画面
            updateWindow();

            // 更新游戏数据
            clearLine();
        }

        if(gameOver){
            // 保存分数
            saveScore();
            // 显示游戏结束界面
            displayOver();

            system("pause");
            init(); // 重新开局
        }
    }

}

void Tetris::updateWindow() {

    // 绘制背景图片
    putimage(0, 0, &imgBg);

    IMAGE **imgs = Block::getImages();
    BeginBatchDraw();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (map[i][j] == 0)continue;

            int x = j * blockSize + leftMargin;
            int y = i * blockSize + topMargin;
            putimage(x, y, imgs[map[i][j] - 1]);
        }
    }

    curBlock->draw(leftMargin, topMargin);
    nextBlock->draw(220, topMargin);

    drawScore(); // 绘制分数

    EndBatchDraw();


}

void Tetris::init() {
    delay = SPEED_NORMAL[0];

//    // 配置随机种子
//    srand(time(NULL));
    // 创建游戏窗口
    initgraph(300, 288);

    // 加载背景图片
    loadimage(&imgBg, "E:\\code\\cpp\\Learn-CPP\\01-Tetris\\res\\bg.png");

    // 初始化游戏区中的数据
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            map[i][j] = 0;
        }
    }

    score = 0;
    lineCount = 0;
    level = 1;

    // 初始化最高分
    ifstream file(RECORDER_FILE);
    if(!file.is_open()){
        cout << RECORDER_FILE << "open fail" << endl;
        highestScore = 0;
    }
    else{
        file >> highestScore;
    }
    file.close();

    gameOver = false;
}

void Tetris::keyEvent() {

    unsigned char ch = ' '; //0..255
    bool rotateFlag = false;
    int dx = 0;
    if (_kbhit()) {
        ch = _getch();

        // 按下方向按键，会自动返回两个字符
        // 如果按下 向上 方向键，会先后返回：224  72
        // 如果按下 向下 方向键，会先后返回：224  80
        // 如果按下 向左 方向键，会先后返回：224  75
        // 如果按下 向右 方向键，会先后返回：224  77
        if (ch == 224) {
            ch = _getch();
            switch (ch) {
                case 72:
                    rotateFlag = true;
                    break;
                case 80:
                    delay = SPEED_FAST;
                    break;
                case 75:
                    dx = -1;
                    break;
                case 77:
                    dx = 1;
                    break;
                default:
                    break;
            }
        }
    }

    if (rotateFlag == true) {
        // 做旋转处理
        rotate();
        update = true;
        rotateFlag = false;
    }
    if (dx != 0) {
        // 实现左右移动
        moveLeftRight(dx);
        update = true;
        dx = 0;
    }


}

// 第一次调用，返回0
// 返回距离上一次调用，间隔了多少ms
int Tetris::getDelay() {
    static unsigned long long lastTime = 0;
    unsigned long long currentTime = GetTickCount();
    if (lastTime == 0) {
        lastTime = currentTime;
        return 0;
    } else {
        int ret = currentTime - lastTime;
        lastTime = currentTime;
        return ret;
    }
}

void Tetris::drop() {

    bakBlock = *curBlock;
    curBlock->drop();

    // 下降过程中，
    if (curBlock->blockInMap(map) == false) {
        // 把这个方块“固化”
        bakBlock.solidify(map);
        delete curBlock;
        curBlock = nextBlock;
        nextBlock = new Block;

        // 检查游戏是否结束
        checkOver();

    }

    delay = SPEED_NORMAL[level-1];
}

void Tetris::clearLine() {
    int lines = 0;
    int k = rows - 1;
    for (int i = rows - 1; i >= 0; i--) {
        // 检查第i行是否满行
        int count = 0;
        for (int j = 0; j < cols; j++) {
            if (map[i][j]) {
                count++;
            }
            map[k][j] = map[i][j];//一边扫描一边存储
        }
        if (count < cols) { // 不是满行
            k--;
        } else { // 满行
            lines++;
        }
    }

    if(lines > 0){
        // 计算得分
        int addScore[4] = {1, 3, 6, 8};
        score += addScore[lines - 1];
        lineCount += lines;
        update = true;

        // 每10分一关，0-10第一关，11-20第二关
        level = (score+9)/10;
        if(level > MAX_LEVEL){
            gameOver = true;
        }
        // 播放音效
        // todo
    }
}

Tetris::Tetris(int rows, int cols, int left, int top, int blockSize) {
    this->rows = rows;
    this->cols = cols;
    this->leftMargin = left;
    this->topMargin = top;
    this->blockSize = blockSize;

    for (int i = 0; i < rows; i++) {
        vector<int> mapRow;
        for (int j = 0; j < cols; j++) {
            mapRow.push_back(0);
        }
        map.push_back(mapRow);
    }
}

void Tetris::moveLeftRight(int offset) {
    bakBlock = *curBlock;
    curBlock->moveLeftRight(offset);

    if (!curBlock->blockInMap(map)) {
        *curBlock = bakBlock;
    }
}

void Tetris::rotate() {
    if (curBlock->getBlockType() == 7) {
        return;
    }
    bakBlock = *curBlock;
    curBlock->rotate();

    if (!curBlock->blockInMap(map)) {
        *curBlock = bakBlock;
    }
}

void Tetris::drawScore() {

    char scoreText[32];
    sprintf_s(scoreText, sizeof(scoreText), "%d", score);
    //string scoreText = to_string(score);

    // 设置字体
    setcolor(RGB(180, 180, 180));

    LOGFONT f;
    gettextstyle(&f); //获取当前的字体
    f.lfHeight = 20;
    f.lfWidth = 10;
    f.lfQuality = ANTIALIASED_QUALITY; // 设置字体为“抗锯齿“效果

    strcpy_s(f.lfFaceName, sizeof(f.lfFaceName), _T("Segoe UI Black"));
    settextstyle(&f);

    setbkmode(TRANSPARENT); //字体的背景设置为透明效果

    outtextxy(215, 235, scoreText);

    char levelText[32];
    sprintf_s(levelText, sizeof(levelText), "%d", level);
    outtextxy(60, 235, levelText);

    char lineCountText[32];
    sprintf_s(lineCountText, sizeof(lineCountText), "%d", lineCount);
    outtextxy(60, 265, lineCountText);

    char highestScoreText[32];
    sprintf_s(highestScoreText, sizeof(highestScoreText), "%d", highestScore);
    outtextxy(215, 265, highestScoreText);
}

void Tetris::checkOver() {

    gameOver = (curBlock->blockInMap(map) == false);

}

void Tetris::saveScore() {
    if(score > highestScore){
        highestScore = score;

        ofstream file(RECORDER_FILE);
        file << highestScore;
        file.close();
    }
}

void Tetris::displayOver() {

    // 可播放音乐

    if(level <= MAX_LEVEL){
        // show game over picture or info
    }
    else{
        // show win piture or info
    }
}
