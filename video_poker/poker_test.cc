#include <iostream>
#include <cassert>
#include <sstream>

#include "poker.h"


void testDraw() {
    Deck d{};
    Card c1 = d.draw();
    assert(c1.suit == Suit::CLUB);
    assert(c1.rank == 2);
    Card c2 = d.draw();
    assert(c2.suit == Suit::CLUB);
    assert(c2.rank == 3);
}

void testShuffle() {
    Deck d{};
    Deck d2{};
    d2.shuffle();
    assert(d != d2);
}

void testCardOstream() {
    std::stringstream ss;
    ss << Card{SPADE, 3};
    assert(ss.str() == "3♠");
    ss.str("");
    ss << Card{HEART, 13};
    assert(ss.str() == "K♥");
}

void testVideoPokerHand() {
    VideoPoker vp{};
    const Hand& hand = vp.deal();
    std::cout << hand << std::endl;
    vp.exchange({true, false, true, true, false});
    std::cout << hand << std::endl;
}

void testHands() {
    VideoPoker vp{};
    Hand max_hand;
    int max_value = 0;
    for (int i = 0; i < 10000000; i++) {
        Hand h = vp.deal();
        if (vp.getHandType(h) >= max_value){
            max_hand = h;
            max_value = vp.getHandType(h);
            std::cout << "Max: " << max_hand << ": " << max_value << ", i:" << i << std::endl;
        }
        h = vp.exchange({true, true, false, false, false});
        // std::cout << h << ": " << vp.getHandType(h) << std::endl;
        if (vp.getHandType(h) >= max_value){
            max_hand = h;
            max_value = vp.getHandType(h);
            std::cout << "Max: " << max_hand << ": " << max_value << ", i:" << i << std::endl;
        }
    }
}

void test_scoring() {
    VideoPoker vp{};
    int total = 0;
    for (int i = 0; i < 1000; i++) {
        Hand h = vp.deal();
        total -= 1;
        h = vp.exchange({true, true, false, false, false});
        int score = vp.score(vp.getHandType(h));
        total += score;
        std::cout << h << "Score: " << score << ", Total: " << total << std::endl;

    }
}

void test_royal_flush() {
    VideoPoker vp {};
    Hand h;
    h[0] = {Suit::CLUB, 10};
    h[1] = {Suit::CLUB, 11};
    h[2] = {Suit::CLUB, 12};
    h[3] = {Suit::CLUB, 13};
    h[4] = {Suit::CLUB, 14};
    std::cout << vp.getHandType(h) << std::endl;
}

void run_tests() {
    // TODO: Add tests for Deck class
    // - Test deck creation (52 cards, no duplicates)
    // - Test shufflegTgT
    // - Test draw

    // TODO: Add tests for VideoPoker class
    testDraw();
    testShuffle();
    testCardOstream();
    test_royal_flush();

    // Output based tests
    // testVideoPokerHand();
    // testHands();
    // test_scoring();
    std::cout << "All tests passed!" << std::endl;
}

int main() {
    run_tests();
    return 0;
}
