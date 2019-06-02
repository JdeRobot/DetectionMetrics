#include "Playback.hpp"

void Playback::onSlide(int currentPos,void *frame){
    *(int *)(frame) = currentPos;
}

Playback::Playback(long long int framesCount):framesCount(framesCount),pause(false),speed(1),frameId(0),inferences(0){
  cv::namedWindow("Detections",cv::WINDOW_NORMAL);
  int *frame = &this->frameId;
  cv::createTrackbar("Frames", "Detections",frame,this->framesCount,&Playback::onSlide,frame);
}

Playback::Playback():framesCount(0),pause(false),speed(1),frameId(0),inferences(0){
  cv::namedWindow("Detections",cv::WINDOW_NORMAL);
}

void Playback::AddTrackbar(long long int framesCount){
  int *frame = &this->frameId;
  this->framesCount = framesCount;
  cv::createTrackbar("Frames", "Detections",frame,this->framesCount,&Playback::onSlide,frame);
}

bool Playback::IsPaused(){
  return this->pause;
}

void Playback::GetInput(char keypressed , cv::Mat inference){
  this->inferences.push_back(inference);
  this->keystroke = keypressed;
  Playback::process();
}

void Playback::GetInput(char keypressed ){
  this->keystroke = keypressed;
  Playback::process();
}

void Playback::process(){
  switch (this->keystroke) {
    case ' ' :
    case 'p' :
    case 'k' : this->pause = !this->pause;
               std::cout << "Keystroke : " << this->keystroke << std::endl;
               break;
    case '+' : this->speed += 2;
               std::cout << "Keystroke : " << this->keystroke << std::endl;
               break;
    case '-' : if(this->speed>2)
                  this->speed-=2;
               std::cout << "Keystroke : " << this->keystroke << std::endl;
               break;
    default  : break;
  }
  Playback::show();
}

void Playback::show(){
  if(!this->pause){
    // Playback::WaitTillResume();
    usleep(int(this->rate()*10000));
    cv::imshow("Detections",this->inferences[this->frameId++]);
    cv::setTrackbarPos("Frames","Detections",this->frameId);
  }
}

void Playback::completeShow(){
  while(this->frameId!=this->inferences.size())
    Playback::GetInput(cv::waitKey(1));
}

double Playback::rate(){
  return this->speed;
}

void Playback::WaitTillResume(){
  while(this->pause)
    Playback::GetInput(cv::waitKey(0));
}

void Playback::updateFrame(int frameId){
  this->frameId = frameId;
}
