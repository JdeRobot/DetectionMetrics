#include "Playback.hpp"

void Playback::onSlide(int currentPos,void *frame){
    *(int *)(frame) = currentPos;
}

Playback::Playback(long long int framesCount):framesCount(framesCount),pause(false),speed(1),frameId(0),inferences(0),groundTruth(0){
  int *frame = &this->frameId;
  cv::namedWindow("Detection",cv::WINDOW_NORMAL);
  cv::namedWindow("GT on RGB",cv::WINDOW_NORMAL);
  cv::createTrackbar("Frames", "Detection",frame,this->framesCount,&Playback::onSlide,frame);
  cv::createTrackbar("Frames", "GT on RGB",frame,this->framesCount,&Playback::onSlide,frame);
  // this->livefeed = new std::queue <cv::Mat>;
  // this->inf_livefeed = new std::queue <cv::Mat>;
}

Playback::Playback():framesCount(0),pause(false),speed(1),frameId(0),inferences(0),groundTruth(0){
  cv::namedWindow("Detection",cv::WINDOW_NORMAL);
  cv::namedWindow("GT on RGB",cv::WINDOW_NORMAL);
  // this->livefeed = new std::queue <cv::Mat>;
  // this->inf_livefeed = new std::queue <cv::Mat>;
}

void Playback::AddTrackbar(long long int framesCount){
  int *frame = &this->frameId;
  this->framesCount = framesCount;
  cv::createTrackbar("Frames", "Detection",frame,this->framesCount,&Playback::onSlide,frame);
  cv::createTrackbar("Frames", "GT on RGB",frame,this->framesCount,&Playback::onSlide,frame);
}

bool Playback::IsPaused(){
  return this->pause;
}

void Playback::GetInput(char keypressed , cv::Mat inference, cv::Mat groundTruth){
  this->inferences.push_back(inference);
  this->groundTruth.push_back(groundTruth);
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
    case '-' : this->speed += 2;
               std::cout << "Keystroke : " << this->keystroke << std::endl;
               break;
    case '+' : if(this->speed>2)
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
    cv::imshow("Detection",this->inferences[this->frameId]);
    cv::imshow("GT on RGB",this->groundTruth[this->frameId]);
    this->frameId++;
    cv::setTrackbarPos("Frames","Detection",this->frameId);
    cv::setTrackbarPos("Frames","GT on RGB",this->frameId);
  }
  else if (this->keystroke=' ')
    Playback::WaitTillResume();
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
