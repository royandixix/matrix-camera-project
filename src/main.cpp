#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>

using namespace cv;
using namespace std;
using namespace chrono;

constexpr const char* RESET    = "\033[0m";
constexpr const char* WHITE    = "\033[97m";
constexpr const char* DIM      = "\033[2m";
constexpr const char* BOLD     = "\033[1m";
constexpr const char* HOME     = "\033[H";
constexpr const char* CLR      = "\033[2J";
constexpr const char* HIDE_CUR = "\033[?25l";
constexpr const char* SHOW_CUR = "\033[?25h";

const string RAMP = " .`'^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhaoe*#MW&8%B@$";

char toAscii(uchar px) {
    return RAMP[px * (RAMP.size() - 1) / 255];
}

string statusBar(double fps, int w, int h) {
    ostringstream ss;
    ss << DIM << " ASCII-CAM  "
       << WHITE << BOLD << w << "x" << h
       << RESET << DIM << "  FPS: "
       << WHITE << static_cast<int>(fps)
       << RESET << DIM << "  [ESC] keluar\n" << RESET;
    return ss.str();
}

string frameToAscii(const Mat& gray) {
    string out;
    out.reserve(gray.rows * (gray.cols + 1));
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j)
            out += toAscii(gray.at<uchar>(i, j));
        out += '\n';
    }
    return out;
}

int main() {
    constexpr int COLS = 160, ROWS = 50;

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[!] Gagal membuka kamera\n";
        return 1;
    }

    cout << HIDE_CUR << CLR;

    auto t0 = steady_clock::now();
    double fps = 0.0;

    while (true) {
        auto t1 = steady_clock::now();
        fps = 1e6 / duration_cast<microseconds>(t1 - t0).count();
        t0 = t1;

        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        flip(frame, frame, 1);
        frame.convertTo(frame, -1, 1.4, 10);
        resize(frame, frame, Size(COLS, ROWS));

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        cout << HOME
             << statusBar(fps, COLS, ROWS)
             << WHITE
             << frameToAscii(gray)
             << RESET;
        cout.flush();

        if (waitKey(1) == 27) break;
    }

    cout << SHOW_CUR << RESET;
    cap.release();
    return 0;
}