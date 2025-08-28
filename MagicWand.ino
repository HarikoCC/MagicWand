#include <Wire.h>
#include <MPU6500_WE.h>
#include <ArduinoJson.h>
#include <ArduinoBLE.h>
#include "MadgWick.h"

#define MPU6500_ADDR 0x68

// 蓝牙配置
BLEService MyService("FFFA");
BLEStringCharacteristic SendData("FFFC", BLENotify, 256);
BLEDevice central;

// 定时器相关变量
hw_timer_t *mytimer = NULL;
int cnt_10ms;
int func_10ms, func_100ms, func_500ms, func_1s;

// 管脚相关
const int SCL_MPU = 5;  // SCK Pin
const int SDA_MPU = 18;  // "MOSI" Pin
const int ADO_MPU = 19;  // "MISO" Pin
const int NCS_MPU = 21;  // Chip Select Pin
const int TRN_KEY = 22;  // Training key
const int INF_KEY = 23;  // Inference key
const int LED = 2;

// 灯和开关标记
int LED_MODE = 0;
int BUT_MODE = 0;
int MODE = 0;
int PRE_BUT = 0;

// SPI标志位
bool SPI_CTRL = true;    // SPI use flag

MPU6500_WE myMPU6500 = MPU6500_WE(&SPI, NCS_MPU, SDA_MPU, ADO_MPU, SCL_MPU, SPI_CTRL);
Madgwick filter;

// 运动轨迹信息
int idx=0;
float raw[2][1000];
float gx_bias, gy_bias, gz_bias;
double vx = 0.0, vz = 0.0;
double x = 0.0, z = 0.0;
double minx = 0.0, minz = 0.0, maxx = 0.0, maxz = 0.0;
int graph[24][24]; // 24*24 轨迹图
double roll, pitch, yaw; // 方位角
int Tx_FLAG = 0, tx_idx=0;

// 输出推理结果
void res_out(int no)
{
  while(no--)
  {
    digitalWrite(LED, HIGH);
    delay(500);
    digitalWrite(LED, LOW);
    delay(500);
  }
}

// 时钟中断 ISR
void timer_isr()
{
    cnt_10ms++;
    func_10ms=1;

    if(cnt_10ms%10==0) func_100ms=1;
    if(cnt_10ms%50==0) func_500ms=1;
    if(cnt_10ms>=100) {cnt_10ms=0; func_1s=1;}
}

// 蓝牙初始化
void BLE_init()
{
  while (!BLE.begin()) {
    Serial.println("starting Bluetooth® Low Energy module failed!");
    delay(100);
  }

  BLE.setLocalName("MagicWand");
  BLE.setAdvertisedService(MyService);

  MyService.addCharacteristic(SendData);
  BLE.addService(MyService);
  BLE.advertise();

  Serial.println("BLE advertising...");
}

// 定时器初始化
void timer_init()
{
  cnt_10ms=0;
  func_10ms=0;
  func_500ms=0;
  func_1s=0;

  mytimer=timerBegin(10000);
  timerAttachInterrupt(mytimer, &timer_isr);
  timerAlarm(mytimer, 100, true, 0);
}

// 加速度计初始化
void mpu_init()
{
  pinMode(TRN_KEY, INPUT_PULLUP);
  pinMode(INF_KEY, INPUT_PULLUP);
  pinMode(LED, OUTPUT);

  Wire.begin();
  while(!myMPU6500.init())
    Serial.println("MPU6500 does not respond");
  Serial.println("MPU6500 is connected");

  myMPU6500.enableGyrDLPF();
  myMPU6500.setGyrDLPF(MPU6500_DLPF_6);
  myMPU6500.setSampleRateDivider(9);
  myMPU6500.setGyrRange(MPU6500_GYRO_RANGE_250);
  myMPU6500.setAccRange(MPU6500_ACC_RANGE_2G);
  myMPU6500.enableAccDLPF(true);
  myMPU6500.setAccDLPF(MPU6500_DLPF_6);

  Serial.println("Position you MPU6500 flat and don't move it - calibrating...");
  delay(1000);
  myMPU6500.autoOffsets();
  Serial.println("Done!");
    float gx_sum = 0, gy_sum = 0, gz_sum = 0;
  const int samples = 100;
  for (int i = 0; i < samples; i++) {
    xyzFloat gyr = myMPU6500.getGyrValues();
    gx_sum += gyr.x;
    gy_sum += gyr.y;
    gz_sum += gyr.z;
    delay(10);
  }
  gx_bias = gx_sum / samples;
  gy_bias = gy_sum / samples;
  gz_bias = gz_sum / samples;
  Serial.printf("Gyro bias: %.3f, %.3f, %.3f\n", gx_bias, gy_bias, gz_bias);
}

// 获取瞬时加速度
void mpu_func()
{
  xyzFloat gValue = myMPU6500.getGValues();
  xyzFloat gyr = myMPU6500.getGyrValues();
  float temp = myMPU6500.getTemperature();
  float resultantG = myMPU6500.getResultantG(gValue);
  track_calc(gValue.x, gValue.y, gValue.z, gyr.x, gyr.y, gyr.z);
}

// 二维轨迹解算：从加速度到地面坐标系位移
void track_calc(float ax_raw, float ay_raw, float az_raw, float gx_raw, float gy_raw, float gz_raw)
{
  filter.updateIMU(
    gx_raw,
    gy_raw,
    gz_raw,
    ax_raw,
    ay_raw,
    az_raw
  );

  Quaternion q = filter.getQuaternion();
  float q1 = q.w;
  float q2 = q.x;
  float q3 = q.y;
  float q4 = q.z;
  double ax_world = ax_raw * (q1*q1 + q2*q2 - q3*q3 - q4*q4) +
                    ay_raw * 2.0*(q2*q3 - q1*q4) +
                    az_raw * 2.0*(q2*q4 + q1*q3);

  double ay_world = ax_raw * 2.0*(q2*q3 + q1*q4) +
                    ay_raw * (q1*q1 - q2*q2 + q3*q3 - q4*q4) +
                    az_raw * 2.0*(q3*q4 - q1*q2);

  double az_world = ax_raw * 2.0*(q2*q4 - q1*q3) +
                    ay_raw * 2.0*(q3*q4 + q1*q2) +
                    az_raw * (q1*q1 - q2*q2 - q3*q3 + q4*q4);

  double ax_motion = ax_world;
  double ay_motion = ay_world;
  double az_motion = az_world - 1.0;

  double acc_mag = sqrt(ax_motion*ax_motion + ay_motion*ay_motion + az_motion*az_motion);
  bool isStationary = (abs(gx_raw) < 0.1 && abs(gy_raw) < 0.1 && abs(gz_raw) < 0.1) && (abs(acc_mag) < 0.15);  // 运动加速度 < 0.15g
  if (isStationary) {
    vx *= 0.0;
    vz *= 0.0;
  }

  if(BUT_MODE==1||BUT_MODE==2)
  {
    const double dt = 0.01;
    vx += ax_motion * dt * 9.8;
    vz += az_motion * dt * 9.8;
    x += vx * dt;
    z += vz * dt;

    if (x < minx) minx = x;
    if (x > maxx) maxx = x;
    if (z < minz) minz = z;
    if (z > maxz) maxz = z;

    raw[0][idx]=x;
    raw[1][idx++]=z;

    StaticJsonDocument<200> doc;
    doc["X="] = x;
    doc["Z="] = z;
    doc["VX="] = vx;
    doc["VZ="] = vz;
    doc["MAXX="]= maxx;
    doc["MINX="]= minx;
    String output;
    serializeJson(doc, output);
    //Serial.println(output);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing...");

  BLE_init();
  mpu_init();
  timer_init();

  float sampleFreq = 100.0; // 采样率 100Hz
  filter.begin(sampleFreq);
}

void loop()
{
  // 检查蓝牙中央设备连接状态
  if (!central.connected()) {
    central = BLE.central();  // 尝试获取新连接
    if (central) {
      Serial.print("Connected to central: ");
      Serial.println(central.address());
    }
  }

  // 每10ms执行一次
  if (func_10ms)
  {
    func_10ms = 0;
    mpu_func();

    // BLE每10ms发送一行，防止影响采样
    if(Tx_FLAG)
    {
      if(tx_idx < 24)
      {
        char temp[25];
        for (int j = 0; j < 24; j++)
          temp[j] = graph[tx_idx][j] ? '1' : '0';
        temp[24] = '\0';
        SendData.setValue(temp);
        LED_MODE = !LED_MODE;
        digitalWrite(LED, LED_MODE);
        tx_idx++;
      }
      if(tx_idx>=24)
      {
        Tx_FLAG=0;
        tx_idx=0;
        SendData.setValue("DATA_END");
        LED_MODE = LOW;
        digitalWrite(LED, LED_MODE);
      }
    }
    else
    {
      // 读取按钮状态
      int trn = !digitalRead(TRN_KEY);
      int inf = !digitalRead(INF_KEY);
      int anybut = trn || inf;
      PRE_BUT = BUT_MODE;

      // 按钮处理
      if (trn && BUT_MODE == 3) // 训练按钮按下，进入训练模式
      {
        for (int i = 0; i < 24; i++)
          for (int j = 0; j < 24; j++)
            graph[i][j] = 0;
        idx = 0;
        x = 0.0;
        z = 0.0;
        vx = 0.0;
        vz = 0.0;
        minx = minz = 1.0 * 1e6;
        maxx = maxz = -1.0 * 1e6;
        BUT_MODE = 1;
      }
      else if (inf && BUT_MODE == 3) // 推理按钮按下，进入推理模式
      {
        for (int i = 0; i < 24; i++)
          for (int j = 0; j < 24; j++)
            graph[i][j] = 0;
        idx = 0;
        x = 0.0;
        z = 0.0;
        vx = 0.0;
        vz = 0.0;
        minx = minz = 1.0 * 1e6;
        maxx = maxz = -1.0 * 1e6;
        BUT_MODE = 2;
      }
      else if (BUT_MODE == 3 && anybut == 0) // 去抖后无按键，进入悬空
      {
        BUT_MODE = 0;
      }

      // 释放按钮，处理数据
      if ((BUT_MODE == 1 || BUT_MODE == 2) && anybut == 0)
      {
        Serial.println(maxx);
        Serial.println(minx);
        Serial.println(maxz);
        Serial.println(minz);

        // 归一化数据并映射到24x24网格
        for (int i = 0; i < idx; i++)
        {
          float grid_x_f = (raw[0][i] - minx) * 23.0f / (maxx - minx);
          float grid_z_f = (raw[1][i] - minz) * 23.0f / (maxz - minz);

          int grid_x = (int)round(grid_x_f);
          int grid_z = 23 - (int)round(grid_z_f);  // Y轴翻转

          grid_x = constrain(grid_x, 0, 23);
          grid_z = constrain(grid_z, 0, 23);

          graph[grid_z][grid_x] = 1;
        }

        // 打印图形数据
        for (int i = 0; i < 24; i++)
        {
          for (int j = 0; j < 23; j++)
            Serial.print(graph[i][j]);
          Serial.println(graph[i][23]);
        }
        Serial.println();
        Serial.println();

        // 重置状态
        BUT_MODE = 0;
        LED_MODE = LOW;
        digitalWrite(LED, LED_MODE);

        // 如果之前是训练模式，发送训练数据
        if (PRE_BUT == 1)
        {
          Tx_FLAG = 1;
          SendData.setValue("DATA_BEGIN");
        }
          
      }

      // 按钮按下且处于悬空状态，进入去抖动状态
      if (anybut == 1 && BUT_MODE == 0)
        BUT_MODE = 3;
    }

    if (func_100ms)
    {
      func_100ms = 0;
      if (BUT_MODE == 1 || BUT_MODE == 2)
      {
        LED_MODE = !LED_MODE;
        digitalWrite(LED, LED_MODE);
      }
    }
  }
}