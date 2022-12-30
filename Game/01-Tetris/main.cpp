/*
 * 1 创建项目
 * 2 导入素材（res文件夹)
 * 3 C++开发
 * 4 设计C++项目的模块（类）
 *  4.1 Block 方块
 *  4.2 Tetris 俄罗斯方块游戏
 * 5 设计各个类的主接口
 * 6 启动游戏
 */
#include <iostream>
#include "Tetris.h"



int main() {
    Tetris game(20, 10, 85, 48, 10);
    game.play();
    return 0;
}
