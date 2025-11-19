#include "deck.h"

#include <random>
#include <iostream>

Hand dealTestHand(Deck& deck) {
    Hand h; 
    for (int i = 0; i < 5; i++) {
        h[i] = deck.draw();
    }
    return h;
}

void testStackFullHouse(std::mt19937& rng) {
    Deck d {rng};
    for (int i = 2; i <= 13; i++) {
        d.stack(FULL_HOUSE, i, i+1);
        Hand h = dealTestHand(d);
        std::cout << "Full House of " << i << "'s: " << h << std::endl;
    }
}

void testStackTwoPair(std::mt19937& rng) {
    Deck d {rng};
    for (int i = 2; i <= 13; i++) {
        d.stack(TWO_PAIR, i, i+1);
        Hand h = dealTestHand(d);
        std::cout << "Two Pair of " << i << "'s: " << h << std::endl;
    }
}

void testStackFourOfAKind(std::mt19937& rng) {
    Deck d {rng};
    for (int i = 2; i <= 14; i++) {
        d.stack(FOUR_OF_A_KIND, i, -1);
        Hand h = dealTestHand(d);
        std::cout << "Four-of-a-kind of " << i << "'s: " << h << std::endl;
    }
}

void testStackThreeOfAKind(std::mt19937& rng) {
    Deck d {rng};
    for (int i = 2; i <= 14; i++) {
        d.stack(THREE_OF_A_KIND, i, -1);
        Hand h = dealTestHand(d);
        std::cout << "Three-of-a-kind of " << i << "'s: " << h << std::endl;
    }
}

void testStackHighPair(std::mt19937& rng) {
    Deck d {rng};
    for (int i = 11; i <= 14; i++) {
        d.stack(HIGH_PAIR, i, -1);
        Hand h = dealTestHand(d);
        std::cout << "High Pair of " << i << "'s: " << h << std::endl;
    }
}

void testStackPair(std::mt19937& rng) {
    Deck d {rng};
    for (int i = 2; i <= 10; i++) {
        d.stack(PAIR, i, -1);
        Hand h = dealTestHand(d);
        std::cout << "Pair of " << i << "'s: " << h << std::endl;
    }
}

void mainDeckTest() {
    std::random_device rd {};
    std::mt19937 rng {rd()};    

    testStackPair(rng);
    testStackHighPair(rng);
    testStackThreeOfAKind(rng);
    testStackFourOfAKind(rng);
    testStackTwoPair(rng);
    testStackFullHouse(rng);
}