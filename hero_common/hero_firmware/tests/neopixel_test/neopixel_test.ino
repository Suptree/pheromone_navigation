 /************************************************************************
 *  H E R O   N E O   P I X E L   T E S T
 ************************************************************************
 *
 * Paulo Rezeck <rezeck@dcc.ufmg.br>
 * Mauricio Ferrari <mauferrari@dcc.ufmg.br>
 * Hector Azpurua <hectorxxx@gmail.com>
 * 
 * HeRo Project
 * Computer Vision and Robotics Lab
 * Federal University of Minas Gerais - Brazil
 ************************************************************************/
/* ROS Library */
#define ROSSERIAL_ARDUINO_TCP
#include <ros.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/ColorRGBA.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

#include <ESP8266WiFi.h>

#include <NeoPixelBus.h>
#include <NeoPixelBrightnessBus.h>
#define colorSaturation 255

const uint16_t PixelCount = 7; // this example assumes 4 pixels, making it smaller will cause a failure
NeoPixelBrightnessBus<NeoGrbFeature, NeoEsp8266Uart1800KbpsMethod> strip(PixelCount);
RgbColor red(colorSaturation, 0, 0);
RgbColor green(0, colorSaturation, 0);
RgbColor black(0, 0, 0);

#define HERO_ID "5"

/* Timer */
double timer, log_timer, rate = 10;

/* Wifi setup */
IPAddress ROS_MASTER_ADDRESS(10, 42, 0, 1); // ros master ip
char* WIFI_SSID = "hero_network"; // network name
char* WIFI_PASSWD = "s3cr3tp4ss"; // network password

/* ROS Setup */
String hero_name;
/* ROS Node Instaciatation */
ros::NodeHandle nh;

/* LED callback */
void led_callback(const std_msgs::ColorRGBA& msg);
String led_topic;
ros::Subscriber<std_msgs::ColorRGBA> *led_sub;

char buf [100];

/* Setup */
void setup() {
  /* Connect the ESP8266 the the wifi AP */
  WiFi.begin(WIFI_SSID, WIFI_PASSWD);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  
  Serial.begin(115200);

  /* Start ROS communication module */
  uint16_t ROS_MASTER_PORT = 11411;
  nh.getHardware()->setConnection(ROS_MASTER_ADDRESS, ROS_MASTER_PORT);

  /* Setup laser publisher */
  hero_name = String("/hero_") + String(HERO_ID);
  
  led_topic = hero_name + String("/led");
  led_sub = new ros::Subscriber<std_msgs::ColorRGBA>(led_topic.c_str(), led_callback);

  /* Starting ros node */
  nh.initNode();
  
  /* Address Subscribers */
  nh.subscribe(*led_sub);

  /* Setup complete esp LED message */
  // this resets all the neopixels to an off state
  strip.Begin();
  strip.Show();
  for(int i = 0; i <= PixelCount; i++){
    strip.SetPixelColor(i, black);
  }
  strip.SetBrightness(64);
  strip.Show();
  
  //pinMode(LED_BUILTIN, OUTPUT);
  for(int i = 0; i <= PixelCount * 6; i++){
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    if (i % 8 > 2) continue;
      strip.SetPixelColor(i % 8, green);
    strip.SetBrightness(16);
    strip.Show();
    delay(100);                   
    strip.SetPixelColor(i % 8, black);
  }
  strip.Show();
  
  /* ROS LOG */
  sprintf(buf,"\33[96m[%s] Hero test mode - NeoPixel Test! \33[0m", hero_name.c_str());
  nh.loginfo(buf);
}



/************************************************************************
 * M A I N  L O 0 P
 ************************************************************************/
void loop() { 
    /* ROS INFOS */
    if ((millis() - log_timer) > 3000){
      log_timer = millis();
      sprintf(buf,"\33[96m[%s] Conected at time %d\33[0m", hero_name.c_str(), millis());
      nh.loginfo(buf);
    }

    /* ROS Loop */
    if (millis() - timer > 1000.0/rate){ 
      timer = millis();
      nh.spinOnce();
      
    }
}

void led_callback(const std_msgs::ColorRGBA& msg){
  //strip.Begin();
  //delay(50);
  RgbColor color(msg.r * 255, msg.g * 255, msg.b * 255);
  for(int i = 0; i <= PixelCount; i++){
    //digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    strip.SetPixelColor(i, color);
  }
  strip.SetBrightness(msg.a * 255);
  strip.Show();
}
