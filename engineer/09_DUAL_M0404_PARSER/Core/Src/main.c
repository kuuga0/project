/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>

/*
 * 这些宏会被后面的类型定义使用，
 * 因此必须放在PTD类型定义之前。
 */
#define M0404_DMA_SIZE           64U
#define M0404_RING_SIZE          512U
#define M0404_FRAME_SIZE         35U
#define M0404_POINT_COUNT        16U

/* TIM6 runs at 200 Hz, so 10 ticks equal a 50 ms pressure-report period. */
#define PRESSURE_REPORT_TICKS     10U
#define DIAGNOSTIC_REPORT_TICKS   200U

#define M0404_HEADER_0           0xAAU
#define M0404_HEADER_1           0x01U


/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

typedef struct
{
    /* ================= 环形缓冲区 ================= */

    uint8_t ring_buffer[M0404_RING_SIZE];

    /*
     * head：接收中断下一次写入的位置
     * tail：主循环下一次读取的位置
     */
    volatile uint16_t ring_head;
    volatile uint16_t ring_tail;

    /* ================= 帧解析器 ================= */

    uint8_t frame_buffer[M0404_FRAME_SIZE];

    /*
     * 0：正在寻找AA
     * 1：已经收到AA，等待01
     * 2～34：继续收集候选帧
     */
    uint16_t frame_index;

    /* 最新一帧的16个原始压力数值 */
    uint16_t pressure[M0404_POINT_COUNT];

    /* ================= 有效数据状态 ================= */

    volatile uint32_t sequence;
    volatile uint32_t last_update_ms;
    volatile uint8_t new_frame;

    /* ================= 接收统计 ================= */

    volatile uint32_t dma_blocks;
    volatile uint32_t total_bytes;
    volatile uint32_t ring_overflow;

    /* ================= 协议错误统计 ================= */

    volatile uint32_t checksum_error;
    volatile uint32_t discarded_bytes;

    volatile uint32_t last_checksum_error_ms;
    volatile uint8_t last_checksum_calculated;
    volatile uint8_t last_checksum_received;

    /* ================= UART与DMA错误 ================= */

    volatile uint32_t restart_error;

    volatile uint32_t uart_error;
    volatile uint32_t last_uart_error;
    volatile uint32_t last_uart_error_ms;

    volatile uint8_t recover_request;

} M0404_Channel_t;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

TIM_HandleTypeDef htim6;

UART_HandleTypeDef huart1;
UART_HandleTypeDef huart3;
DMA_HandleTypeDef hdma_usart1_tx;
DMA_HandleTypeDef hdma_usart1_rx;
DMA_HandleTypeDef hdma_usart3_rx;

/* USER CODE BEGIN PV */

/* ================= 200 Hz系统节拍 ================= */

volatile uint32_t timer_200hz_count = 0U;
volatile uint8_t pressure_report_flag = 0U;
volatile uint8_t diagnostic_report_flag = 0U;


/* ================= 两块M0404通道对象 ================= */

/*
 * 初始化为0：
 * 两套环形缓冲区、解析器和统计量全部独立。
 */
M0404_Channel_t m0404_1 = {0};
M0404_Channel_t m0404_2 = {0};


/* ================= 两个DMA接收区 ================= */

/*
 * DMA直接写入这两个数组。
 * STM32H7后续开启D-Cache时，32字节对齐更方便。
 */
__attribute__((aligned(32)))
uint8_t m0404_1_dma_buffer[M0404_DMA_SIZE];

__attribute__((aligned(32)))
uint8_t m0404_2_dma_buffer[M0404_DMA_SIZE];


/* ================= Terminal发送 ================= */

volatile uint8_t uart_tx_busy = 0U;
volatile uint32_t uart_tx_skip_count = 0U;

/*
 * 32点网格和诊断信息较长，
 * 将发送缓冲区扩大到1024字节。
 */
__attribute__((aligned(32)))
uint8_t uart_tx_buffer[1024];

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_TIM6_Init(void);
static void MX_USART3_UART_Init(void);
/* USER CODE BEGIN PFP */

static HAL_StatusTypeDef M0404_StartReceiveDMA(
    UART_HandleTypeDef *huart,
    uint8_t *dma_buffer,
    uint16_t buffer_size
);

static void M0404_RingWrite(
    M0404_Channel_t *channel,
    const uint8_t *data,
    uint16_t length
);

static uint8_t M0404_RingReadByte(
    M0404_Channel_t *channel,
    uint8_t *byte
);

static uint8_t M0404_CalculateChecksum(
    const uint8_t *frame
);

static void M0404_AcceptFrame(
    M0404_Channel_t *channel
);

static void M0404_ResyncAfterBadFrame(
    M0404_Channel_t *channel
);

static void M0404_FeedByte(
    M0404_Channel_t *channel,
    uint8_t byte
);

static void M0404_ProcessRing(
    M0404_Channel_t *channel
);

static void M0404_RecoverIfNeeded(
    M0404_Channel_t *channel,
    UART_HandleTypeDef *huart,
    uint8_t *dma_buffer
);

static void UART1_TrySendPressureFrame(void);
static void UART1_TrySendDiagnosticReport(void);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

static HAL_StatusTypeDef M0404_StartReceiveDMA(
    UART_HandleTypeDef *huart,
    uint8_t *dma_buffer,
    uint16_t buffer_size
)
{
    HAL_StatusTypeDef status;

    status = HAL_UARTEx_ReceiveToIdle_DMA(
        huart,
        dma_buffer,
        buffer_size
    );

    if (status == HAL_OK)
    {
        /*
         * 不使用DMA半传输回调。
         * 只处理：
         * 1. UART空闲事件
         * 2. DMA缓冲区收满事件
         */
        if (huart->hdmarx != NULL)
        {
            __HAL_DMA_DISABLE_IT(
                huart->hdmarx,
                DMA_IT_HT
            );
        }
    }

    return status;
}

static void M0404_RingWrite(
    M0404_Channel_t *channel,
    const uint8_t *data,
    uint16_t length
)
{
    for (uint16_t i = 0U; i < length; i++)
    {
        uint16_t next_head =
            channel->ring_head + 1U;

        if (next_head >= M0404_RING_SIZE)
        {
            next_head = 0U;
        }

        /*
         * next_head追上tail，表示环形缓冲区已满。
         * 当前选择丢弃新字节，不覆盖尚未处理的旧字节。
         */
        if (next_head == channel->ring_tail)
        {
            channel->ring_overflow++;
        }
        else
        {
            channel->ring_buffer[
                channel->ring_head
            ] = data[i];

            channel->ring_head = next_head;
        }
    }
}

static uint8_t M0404_RingReadByte(
    M0404_Channel_t *channel,
    uint8_t *byte
)
{
    /*
     * head等于tail表示缓冲区为空。
     */
    if (channel->ring_head ==
        channel->ring_tail)
    {
        return 0U;
    }

    *byte = channel->ring_buffer[
        channel->ring_tail
    ];

    channel->ring_tail++;

    if (channel->ring_tail >=
        M0404_RING_SIZE)
    {
        channel->ring_tail = 0U;
    }

    return 1U;
}

static uint8_t M0404_CalculateChecksum(
    const uint8_t *frame
)
{
    uint16_t sum = 0U;

    /*
     * 最后一字节frame[34]是接收到的校验值，
     * 所以只累加frame[0]到frame[33]。
     */
    for (uint16_t i = 0U;
         i < (M0404_FRAME_SIZE - 1U);
         i++)
    {
        sum = (uint16_t)(sum + frame[i]);
    }

    /*
     * 强制转换为uint8_t，
     * 等价于只保留低8位。
     */
    return (uint8_t)(sum & 0x00FFU);
}

static void M0404_AcceptFrame(
    M0404_Channel_t *channel
)
{
    for (uint16_t i = 0U;
         i < M0404_POINT_COUNT;
         i++)
    {
        uint16_t high_index =
            2U + i * 2U;

        uint16_t low_index =
            high_index + 1U;

        channel->pressure[i] =
            ((uint16_t)
                channel->frame_buffer[high_index]
                << 8)
            |
            (uint16_t)
                channel->frame_buffer[low_index];
    }

    channel->sequence++;

    channel->last_update_ms =
        HAL_GetTick();

    channel->new_frame = 1U;
}

static void M0404_ResyncAfterBadFrame(
    M0404_Channel_t *channel
)
{
    /*
     * 放弃当前错误候选帧，
     * 从后续数据重新寻找AA 01。
     */
    channel->frame_index = 0U;

    /*
     * 如果错误候选帧最后一个字节是AA，
     * 它可能是下一帧帧头的第一个字节。
     */
    if (channel->frame_buffer[
            M0404_FRAME_SIZE - 1U
        ] == M0404_HEADER_0)
    {
        channel->frame_buffer[0] =
            M0404_HEADER_0;

        channel->frame_index = 1U;
    }
}

static void M0404_FeedByte(
    M0404_Channel_t *channel,
    uint8_t byte
)
{
    /*
     * 状态0：等待第一个帧头字节AA。
     */
    if (channel->frame_index == 0U)
    {
        if (byte == M0404_HEADER_0)
        {
            channel->frame_buffer[0] =
                byte;

            channel->frame_index = 1U;
        }
        else
        {
            channel->discarded_bytes++;
        }

        return;
    }

    /*
     * 状态1：已经收到AA，等待01。
     */
    if (channel->frame_index == 1U)
    {
        if (byte == M0404_HEADER_1)
        {
            channel->frame_buffer[1] =
                byte;

            channel->frame_index = 2U;
        }
        else if (byte == M0404_HEADER_0)
        {
            /*
             * 收到AA AA。
             * 后一个AA仍可能是新帧的开头。
             */
            channel->frame_buffer[0] =
                M0404_HEADER_0;

            channel->frame_index = 1U;
        }
        else
        {
            channel->frame_index = 0U;
            channel->discarded_bytes++;
        }

        return;
    }

    /*
     * 帧头已经确认，继续收集剩余字节。
     */
    channel->frame_buffer[
        channel->frame_index
    ] = byte;

    channel->frame_index++;

    /*
     * 已经收满35字节。
     */
    if (channel->frame_index >=
        M0404_FRAME_SIZE)
    {
        uint8_t calculated_checksum =
            M0404_CalculateChecksum(
                channel->frame_buffer
            );

        uint8_t received_checksum =
            channel->frame_buffer[
                M0404_FRAME_SIZE - 1U
            ];

        if (calculated_checksum ==
            received_checksum)
        {
            M0404_AcceptFrame(channel);

            channel->frame_index = 0U;
        }
        else
        {
            channel->checksum_error++;

            channel->last_checksum_error_ms =
                HAL_GetTick();

            channel->last_checksum_calculated =
                calculated_checksum;

            channel->last_checksum_received =
                received_checksum;

            M0404_ResyncAfterBadFrame(
                channel
            );
        }
    }
}

static void M0404_ProcessRing(
    M0404_Channel_t *channel
)
{
    uint8_t byte = 0U;

    while (M0404_RingReadByte(
               channel,
               &byte
           ) != 0U)
    {
        M0404_FeedByte(
            channel,
            byte
        );
    }
}

static void M0404_RecoverIfNeeded(
    M0404_Channel_t *channel,
    UART_HandleTypeDef *huart,
    uint8_t *dma_buffer
)
{
    if (channel->recover_request == 0U)
    {
        return;
    }

    /*
     * 停止当前异常的RX任务。
     * 只停止接收，不影响USART1发送Terminal信息。
     */
    HAL_UART_AbortReceive(huart);

    /*
     * 清除ORE和接收寄存器残留数据。
     */
    __HAL_UART_CLEAR_OREFLAG(huart);

    __HAL_UART_SEND_REQ(
        huart,
        UART_RXDATA_FLUSH_REQUEST
    );

    /*
     * UART错误意味着字节流可能已经不连续。
     * 放弃当前待处理字节和未完成候选帧，
     * 从下一组AA 01重新同步。
     */
    channel->ring_tail =
        channel->ring_head;

    channel->frame_index = 0U;

    if (M0404_StartReceiveDMA(
            huart,
            dma_buffer,
            M0404_DMA_SIZE
        ) == HAL_OK)
    {
        channel->recover_request = 0U;
    }
    else
    {
        channel->restart_error++;
    }
}

static void UART1_TrySendPressureFrame(void)
{
    if (uart_tx_busy != 0U)
    {
        uart_tx_skip_count++;
        return;
    }

    uint32_t timestamp_ms = HAL_GetTick();

    int length = snprintf(
        (char *)uart_tx_buffer,
        sizeof(uart_tx_buffer),

        "P32,t=%lu,%u,%u,%u,%u,%u,%u,%u,%u,"
        "%u,%u,%u,%u,%u,%u,%u,%u,"
        "%u,%u,%u,%u,%u,%u,%u,%u,"
        "%u,%u,%u,%u,%u,%u,%u,%u\r\n",

        (unsigned long)timestamp_ms,

        (unsigned int)m0404_1.pressure[0],
        (unsigned int)m0404_1.pressure[1],
        (unsigned int)m0404_1.pressure[2],
        (unsigned int)m0404_1.pressure[3],
        (unsigned int)m0404_2.pressure[0],
        (unsigned int)m0404_2.pressure[1],
        (unsigned int)m0404_2.pressure[2],
        (unsigned int)m0404_2.pressure[3],

        (unsigned int)m0404_1.pressure[4],
        (unsigned int)m0404_1.pressure[5],
        (unsigned int)m0404_1.pressure[6],
        (unsigned int)m0404_1.pressure[7],
        (unsigned int)m0404_2.pressure[4],
        (unsigned int)m0404_2.pressure[5],
        (unsigned int)m0404_2.pressure[6],
        (unsigned int)m0404_2.pressure[7],

        (unsigned int)m0404_1.pressure[8],
        (unsigned int)m0404_1.pressure[9],
        (unsigned int)m0404_1.pressure[10],
        (unsigned int)m0404_1.pressure[11],
        (unsigned int)m0404_2.pressure[8],
        (unsigned int)m0404_2.pressure[9],
        (unsigned int)m0404_2.pressure[10],
        (unsigned int)m0404_2.pressure[11],

        (unsigned int)m0404_1.pressure[12],
        (unsigned int)m0404_1.pressure[13],
        (unsigned int)m0404_1.pressure[14],
        (unsigned int)m0404_1.pressure[15],
        (unsigned int)m0404_2.pressure[12],
        (unsigned int)m0404_2.pressure[13],
        (unsigned int)m0404_2.pressure[14],
        (unsigned int)m0404_2.pressure[15]
    );

    if ((length > 0) &&
        (length < (int)sizeof(uart_tx_buffer)))
    {
        uart_tx_busy = 1U;

        if (HAL_UART_Transmit_DMA(
                &huart1,
                uart_tx_buffer,
                (uint16_t)length
            ) != HAL_OK)
        {
            uart_tx_busy = 0U;
        }
    }
}

static void UART1_TrySendDiagnosticReport(void)
{
    if (uart_tx_busy != 0U)
    {
        uart_tx_skip_count++;
        return;
    }

    uint32_t now_ms = HAL_GetTick();

    uint32_t m1_age_ms = 0U;
    uint32_t m2_age_ms = 0U;

    if (m0404_1.sequence != 0U)
    {
        m1_age_ms =
            now_ms - m0404_1.last_update_ms;
    }

    if (m0404_2.sequence != 0U)
    {
        m2_age_ms =
            now_ms - m0404_2.last_update_ms;
    }

    int length = snprintf(
        (char *)uart_tx_buffer,
        sizeof(uart_tx_buffer),

        "M1 seq=%lu, age=%lu ms, ck=%lu, "
        "dma=%lu, bytes=%lu, ovf=%lu, "
        "uart=%lu(0x%08lX), restart=%lu\r\n"

        "M2 seq=%lu, age=%lu ms, ck=%lu, "
        "dma=%lu, bytes=%lu, ovf=%lu, "
        "uart=%lu(0x%08lX), restart=%lu\r\n"

        /* 供PC上位机解析的单行32点数据，顺序与下方4x8网格一致。 */
        "P32DIAG=%u,%u,%u,%u,%u,%u,%u,%u,"
        "%u,%u,%u,%u,%u,%u,%u,%u,"
        "%u,%u,%u,%u,%u,%u,%u,%u,"
        "%u,%u,%u,%u,%u,%u,%u,%u\r\n"

        /* M1和M2均为4x4，物理上左右拼接为4x8。 */
        "GRID4x8 layout=M1|M2\r\n"
        "R0=[%5u,%5u,%5u,%5u | %5u,%5u,%5u,%5u]\r\n"
        "R1=[%5u,%5u,%5u,%5u | %5u,%5u,%5u,%5u]\r\n"
        "R2=[%5u,%5u,%5u,%5u | %5u,%5u,%5u,%5u]\r\n"
        "R3=[%5u,%5u,%5u,%5u | %5u,%5u,%5u,%5u]\r\n",

        /* M1状态 */
        (unsigned long)m0404_1.sequence,
        (unsigned long)m1_age_ms,
        (unsigned long)m0404_1.checksum_error,
        (unsigned long)m0404_1.dma_blocks,
        (unsigned long)m0404_1.total_bytes,
        (unsigned long)m0404_1.ring_overflow,
        (unsigned long)m0404_1.uart_error,
        (unsigned long)m0404_1.last_uart_error,
        (unsigned long)m0404_1.restart_error,

        /* M2状态 */
        (unsigned long)m0404_2.sequence,
        (unsigned long)m2_age_ms,
        (unsigned long)m0404_2.checksum_error,
        (unsigned long)m0404_2.dma_blocks,
        (unsigned long)m0404_2.total_bytes,
        (unsigned long)m0404_2.ring_overflow,
        (unsigned long)m0404_2.uart_error,
        (unsigned long)m0404_2.last_uart_error,
        (unsigned long)m0404_2.restart_error,

        /* P32：每行先输出M1的4点，再输出M2的4点。 */
        (unsigned int)m0404_1.pressure[0],
        (unsigned int)m0404_1.pressure[1],
        (unsigned int)m0404_1.pressure[2],
        (unsigned int)m0404_1.pressure[3],
        (unsigned int)m0404_2.pressure[0],
        (unsigned int)m0404_2.pressure[1],
        (unsigned int)m0404_2.pressure[2],
        (unsigned int)m0404_2.pressure[3],

        (unsigned int)m0404_1.pressure[4],
        (unsigned int)m0404_1.pressure[5],
        (unsigned int)m0404_1.pressure[6],
        (unsigned int)m0404_1.pressure[7],
        (unsigned int)m0404_2.pressure[4],
        (unsigned int)m0404_2.pressure[5],
        (unsigned int)m0404_2.pressure[6],
        (unsigned int)m0404_2.pressure[7],

        (unsigned int)m0404_1.pressure[8],
        (unsigned int)m0404_1.pressure[9],
        (unsigned int)m0404_1.pressure[10],
        (unsigned int)m0404_1.pressure[11],
        (unsigned int)m0404_2.pressure[8],
        (unsigned int)m0404_2.pressure[9],
        (unsigned int)m0404_2.pressure[10],
        (unsigned int)m0404_2.pressure[11],

        (unsigned int)m0404_1.pressure[12],
        (unsigned int)m0404_1.pressure[13],
        (unsigned int)m0404_1.pressure[14],
        (unsigned int)m0404_1.pressure[15],
        (unsigned int)m0404_2.pressure[12],
        (unsigned int)m0404_2.pressure[13],
        (unsigned int)m0404_2.pressure[14],
        (unsigned int)m0404_2.pressure[15],

        /* 32点可视化：每行先打印M1的4点，再打印M2的4点。 */
        (unsigned int)m0404_1.pressure[0],
        (unsigned int)m0404_1.pressure[1],
        (unsigned int)m0404_1.pressure[2],
        (unsigned int)m0404_1.pressure[3],
        (unsigned int)m0404_2.pressure[0],
        (unsigned int)m0404_2.pressure[1],
        (unsigned int)m0404_2.pressure[2],
        (unsigned int)m0404_2.pressure[3],

        (unsigned int)m0404_1.pressure[4],
        (unsigned int)m0404_1.pressure[5],
        (unsigned int)m0404_1.pressure[6],
        (unsigned int)m0404_1.pressure[7],
        (unsigned int)m0404_2.pressure[4],
        (unsigned int)m0404_2.pressure[5],
        (unsigned int)m0404_2.pressure[6],
        (unsigned int)m0404_2.pressure[7],

        (unsigned int)m0404_1.pressure[8],
        (unsigned int)m0404_1.pressure[9],
        (unsigned int)m0404_1.pressure[10],
        (unsigned int)m0404_1.pressure[11],
        (unsigned int)m0404_2.pressure[8],
        (unsigned int)m0404_2.pressure[9],
        (unsigned int)m0404_2.pressure[10],
        (unsigned int)m0404_2.pressure[11],

        (unsigned int)m0404_1.pressure[12],
        (unsigned int)m0404_1.pressure[13],
        (unsigned int)m0404_1.pressure[14],
        (unsigned int)m0404_1.pressure[15],
        (unsigned int)m0404_2.pressure[12],
        (unsigned int)m0404_2.pressure[13],
        (unsigned int)m0404_2.pressure[14],
        (unsigned int)m0404_2.pressure[15]
    );

    if ((length > 0) &&
        (length < (int)sizeof(uart_tx_buffer)))
    {
        uart_tx_busy = 1U;

        HAL_StatusTypeDef status =
            HAL_UART_Transmit_DMA(
                &huart1,
                uart_tx_buffer,
                (uint16_t)length
            );

        if (status != HAL_OK)
        {
            uart_tx_busy = 0U;
        }
    }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART1_UART_Init();
  MX_USART3_UART_Init();
  MX_TIM6_Init();
  /* USER CODE BEGIN 2 */

  /* ================= 启动第一块M0404 ================= */

  __HAL_UART_CLEAR_OREFLAG(&huart1);

  __HAL_UART_SEND_REQ(
      &huart1,
      UART_RXDATA_FLUSH_REQUEST
  );

  if (M0404_StartReceiveDMA(
          &huart1,
          m0404_1_dma_buffer,
          M0404_DMA_SIZE
      ) != HAL_OK)
  {
      Error_Handler();
  }


  /* ================= 启动第二块M0404 ================= */

  __HAL_UART_CLEAR_OREFLAG(&huart3);

  __HAL_UART_SEND_REQ(
      &huart3,
      UART_RXDATA_FLUSH_REQUEST
  );

  if (M0404_StartReceiveDMA(
          &huart3,
          m0404_2_dma_buffer,
          M0404_DMA_SIZE
      ) != HAL_OK)
  {
      Error_Handler();
  }


  /* ================= 启动200 Hz定时器 ================= */

  if (HAL_TIM_Base_Start_IT(&htim6)
      != HAL_OK)
  {
      Error_Handler();
  }


  /*
   * 两路RX DMA已经运行后，
   * 再发送一次启动提示。
   */
  static const uint8_t startup_message[] =
      "\r\nDual M0404 parser started.\r\n";

  HAL_UART_Transmit(
      &huart1,
      (uint8_t *)startup_message,
      sizeof(startup_message) - 1U,
      200
  );

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

	    /*
	     * 每一路只恢复自己的UART和解析状态。
	     */
	    M0404_RecoverIfNeeded(
	        &m0404_1,
	        &huart1,
	        m0404_1_dma_buffer
	    );

	    M0404_RecoverIfNeeded(
	        &m0404_2,
	        &huart3,
	        m0404_2_dma_buffer
	    );

	    /*
	     * 两个环形缓冲区分别解析。
	     */
	    M0404_ProcessRing(&m0404_1);
	    M0404_ProcessRing(&m0404_2);

	    /* 每50 ms发送一帧精简P32数据，供网页热力图实时刷新。 */
	    if (pressure_report_flag != 0U)
	    {
	        pressure_report_flag = 0U;

	        UART1_TrySendPressureFrame();

	        m0404_1.new_frame = 0U;
	        m0404_2.new_frame = 0U;
	    }

	    /* 诊断信息只在DMA空闲时每秒发送一次。 */
	    if ((diagnostic_report_flag != 0U) &&
	        (uart_tx_busy == 0U))
	    {
	        diagnostic_report_flag = 0U;

	        UART1_TrySendDiagnosticReport();
	    }


  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM6 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM6_Init(void)
{

  /* USER CODE BEGIN TIM6_Init 0 */

  /* USER CODE END TIM6_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM6_Init 1 */

  /* USER CODE END TIM6_Init 1 */
  htim6.Instance = TIM6;
  htim6.Init.Prescaler = 63999;
  htim6.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim6.Init.Period = 4;
  htim6.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim6) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim6, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM6_Init 2 */

  /* USER CODE END TIM6_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream0_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream0_IRQn);
  /* DMA1_Stream1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream1_IRQn);
  /* DMA1_Stream2_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream2_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream2_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LED_0_GPIO_Port, LED_0_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin : LED_0_Pin */
  GPIO_InitStruct.Pin = LED_0_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LED_0_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void HAL_TIM_PeriodElapsedCallback(
    TIM_HandleTypeDef *htim
)
{
    if (htim->Instance == TIM6)
    {
        timer_200hz_count++;

        if ((timer_200hz_count % PRESSURE_REPORT_TICKS)
            == 0U)
        {
            pressure_report_flag = 1U;
        }

        if ((timer_200hz_count % DIAGNOSTIC_REPORT_TICKS)
            == 0U)
        {
            diagnostic_report_flag = 1U;
        }
    }
}

void HAL_UARTEx_RxEventCallback(
    UART_HandleTypeDef *huart,
    uint16_t Size
)
{
    /*
     * 第一块M0404：USART1
     */
    if (huart->Instance == USART1)
    {
        if (Size > M0404_DMA_SIZE)
        {
            Size = M0404_DMA_SIZE;
        }

        M0404_RingWrite(
            &m0404_1,
            m0404_1_dma_buffer,
            Size
        );

        m0404_1.dma_blocks++;
        m0404_1.total_bytes += Size;

        if (M0404_StartReceiveDMA(
                &huart1,
                m0404_1_dma_buffer,
                M0404_DMA_SIZE
            ) != HAL_OK)
        {
            m0404_1.restart_error++;
            m0404_1.recover_request = 1U;
        }
    }

    /*
     * 第二块M0404：USART3
     */
    else if (huart->Instance == USART3)
    {
        if (Size > M0404_DMA_SIZE)
        {
            Size = M0404_DMA_SIZE;
        }

        M0404_RingWrite(
            &m0404_2,
            m0404_2_dma_buffer,
            Size
        );

        m0404_2.dma_blocks++;
        m0404_2.total_bytes += Size;

        if (M0404_StartReceiveDMA(
                &huart3,
                m0404_2_dma_buffer,
                M0404_DMA_SIZE
            ) != HAL_OK)
        {
            m0404_2.restart_error++;
            m0404_2.recover_request = 1U;
        }
    }
}

void HAL_UART_TxCpltCallback(
    UART_HandleTypeDef *huart
)
{
    if (huart->Instance == USART1)
    {
        uart_tx_busy = 0U;
    }
}

void HAL_UART_ErrorCallback(
    UART_HandleTypeDef *huart
)
{
    M0404_Channel_t *channel = NULL;

    if (huart->Instance == USART1)
    {
        channel = &m0404_1;
    }
    else if (huart->Instance == USART3)
    {
        channel = &m0404_2;
    }

    if (channel != NULL)
    {
        channel->last_uart_error =
            HAL_UART_GetError(huart);

        /*
         * 主循环尚未处理前，不重复统计同一个错误。
         */
        if (channel->recover_request == 0U)
        {
            channel->uart_error++;

            channel->last_uart_error_ms =
                HAL_GetTick();
        }

        channel->recover_request = 1U;
    }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
