// Player.cpp
#include "Player.h"

// Constructor
Player::Player(const std::string& playerName, int startingMoney)
    : name(playerName), money(startingMoney) {}

// Getters
std::string Player::getName() const
{
    return name;
}
int Player::getMoney() const
{
    return money;
}

// Setters / Game Logic
void Player::addMoney(int amount)
{
    money += amount;
}
void Player::deductMoney(int amount)
{
    money -= amount;
}
void Player::setMoney(int newAmount)
{
    money = newAmount;
}
