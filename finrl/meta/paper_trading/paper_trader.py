# Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Many platforms exist for simulated trading (paper trading) which can be used for building and developing the methods discussed. Please use common sense and always first consult a professional before trading or investing.
# Setup Alpaca Paper trading environment
from __future__ import annotations

import datetime
import threading
import time
import logbook

import alpaca_trade_api as tradeapi
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from abc import ABC, abstractmethod
from datetime import timedelta

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.meta.paper_trading.futu import PaperTradingFutu
from elegantrl.agents import AgentDDPG, AgentTD3, AgentSAC, AgentPPO, AgentA2C
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
MODELS_SB3 = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC}

MODELS = {"ddpg": AgentDDPG, "td3": AgentTD3, "sac": AgentSAC, "ppo": AgentPPO, "a2c": AgentA2C}

class PaperTrader:
    def __init__(
        self,
        ticker_list,
        time_interval,
        drl_lib,
        agent,
        cwd,
        net_dim,
        state_dim,
        action_dim,
        tech_indicator_list,
        turbulence_thresh=30,
        max_stock=1e2,
        latency=None,
        broker="alpaca",
        argv=None
    ):
        # load agent
        self.drl_lib = drl_lib
        self.logger = logbook.Logger(self.__class__.__name__)
        self.logger.info ( f"ticker_list: {ticker_list} time_interval: {time_interval} drl_lib: {drl_lib} agent: {agent} cwd: {cwd} net_dim: {net_dim} state_dim: {state_dim} action_dim: {action_dim}")


        # Load agent for Elegantrl
        if drl_lib == "elegantrl":
            agent_class = MODELS.get(agent)
            if agent_class is None:
                raise ValueError(f"Agent {agent} is not supported for Elegantrl.")
            agent_instance = agent_class(net_dim, state_dim, action_dim)
            try:
                agent_instance.save_or_load_agent(cwd=cwd, if_save=False)
                self.act = agent_instance.act
                self.device = agent_instance.device
                self.logger.info(f"Loaded Elegantrl agent: {agent} on device: {self.device}")
            except (FileNotFoundError, RuntimeError) as e:
                raise ValueError(f"Failed to load Elegantrl agent: {e}") from e
        # Load agent for RLLib
        elif drl_lib == "rllib":
            try:
                from ray.rllib.agents import ppo
                from ray.rllib.agents.ppo import PPOTrainer

                config = ppo.DEFAULT_CONFIG.copy()
                config["env"] = StockEnvEmpty
                config["log_level"] = "WARN"
                config["env_config"] = {
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                }
                trainer = PPOTrainer(env=StockEnvEmpty, config=config)
                trainer.restore(cwd)
                self.agent = trainer
                self.logger.info(f"Restored RLLib agent from checkpoint {cwd}")
            except (FileNotFoundError, ValueError) as e:
                raise ValueError(f"Failed to load RLLib agent: {e}") from e

        # Load agent for Stable Baselines 3
        elif drl_lib == "stable_baselines3":
            try:
                agent_class = MODELS_SB3.get(agent)
                if agent_class is None:
                    raise ValueError(f"Agent {agent} is not supported for Stable Baselines 3.")
                
                self.model = agent_class.load(cwd)
                self.logger.info(f"Successfully loaded Stable Baselines 3 agent from {cwd}")
            except (FileNotFoundError, ValueError) as e:
                raise ValueError(f"Failed to load Stable Baselines 3 agent: {e}") from e


        else:
            raise ValueError(
                f"The DRL library '{drl_lib}' is not supported. Please check your input."
            )

        self.logger.info ( 'start broker init')
        try:
            if ( broker == "alpaca"):
                self.broker = PaperTradingAlpaca(
                    alpaca_api_key=argv["ALPACA_API_KEY"], 
                    alpaca_api_secret=argv["ALPACA_API_SECRET"],
                    alpaca_api_base_url=argv["ALPACA_API_BASE_URL"])
            elif ( broker == "futu"):
                print ( f" argv: {argv}")
                self.broker = PaperTradingFutu(
                    host = argv["FUTU_HOST"],
                    port = argv["FUTU_PORT"],
                    pwd_unlock = argv["FUTU_PWD_UNLOCK"],
                    rsa_file = argv["FUTU_RSA_FILE"],
                    exchange = argv["EXCHANGE"] or "XNYS",
                )
            else:
                raise Exception("Broker input is NOT supported yet.")
        except Exception as e:
            self.logger.error ( e)
            raise Exception(f"Fail to connect broker: {broker}. Please check account info and internet connection. {e}")

        


        # connect to Alpaca trading API
        # try:
        #     self.broker = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
        # except:
        #     raise ValueError(
        #         "Fail to connect Alpaca. Please check account info and internet connection."
        #     )

        # read trading time interval
        self.time_interval = time_interval

        # read trading settings
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_thresh = turbulence_thresh
        self.max_stock = max_stock

        # initialize account
        self.stocks = np.asarray([0] * len(ticker_list))  # stocks holding
        self.stocks_cd = np.zeros_like(self.stocks)
        self.cash = None  # cash record
        self.stocks_df = pd.DataFrame(
            self.stocks, columns=["stocks"], index=ticker_list
        )
        self.asset_list = []
        self.price = np.asarray([0] * len(ticker_list))
        self.stockUniverse = ticker_list
        self.turbulence_bool = 0
        self.equities = []

    def _time_interval_to_sleep(self):
        if self.time_interval == "1Min":
            return 60
        elif self.time_interval == "5Min":
            return 60 * 5
        elif self.time_interval == "15Min":
            return 60 * 15
        elif self.time_interval == "1H":
            return 60 * 60
        elif self.time_interval == "1D":
            return 60 * 60 * 24
        else:
            raise ValueError("The time interval is NOT supported yet. Please check your input.")

    def test_latency(self, test_times=10):
        total_time = 0
        for i in range(0, test_times):
            time0 = time.time()
            self.get_state()
            time1 = time.time()
            temp_time = time1 - time0
            total_time += temp_time
        latency = total_time / test_times
        self.logger.info(f"latency for data processing: {latency}")
        return latency

    def run(self):
        orders = self.broker.list_orders(status="open")
        for order in orders:
            print ( order)
            # self.broker.cancel_order(order.id)

        # Wait for market to open.
        self.logger.info("Waiting for market to open...")
        self.awaitMarketOpen()
        self.logger.info("Market opened.")
        while True:
            # Figure out when the market will close so we can prepare to sell beforehand.
            clock = self.broker.get_clock()
            closingTime = clock.next_close.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()

            self.trade()
            if not isinstance(self.broker.get_account().last_equity, float):
                last_equity = float(self.broker.get_account().last_equity)
            else:
                last_equity = self.broker.get_account().last_equity

            cur_time = time.time()
            self.equities.append([cur_time, last_equity])
            time.sleep(self._time_interval_to_sleep())
            
            
            
            # currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            # self.timeToClose = closingTime - currTime
            # if self.timeToClose < (60 * 2):
            #     # Close all positions when 2 minutes til market close.  Any less and it will be in danger of not closing positions in time.

            #     self.logger.info("Market closing soon.  Closing positions.")

            #     threads = []
            #     positions = self.broker.list_positions()
            #     for position in positions:
            #         if position.side == "long":
            #             orderSide = "sell"
            #         else:
            #             orderSide = "buy"
            #         qty = abs(int(float(position.qty)))
            #         respSO = []
            #         tSubmitOrder = threading.Thread(
            #             target=self.submitOrder(qty, position.symbol, orderSide, respSO)
            #         )
            #         tSubmitOrder.start()
            #         threads.append(tSubmitOrder)  # record thread for joining later

            #     for x in threads:  #  wait for all threads to complete
            #         x.join()

            #     # Run script again after market close for next trading day.
            #     self.logger.info("Sleeping until market close (15 minutes).")
            #     time.sleep(60 * 15)

            # else:
            #     self.trade()
            #     if not isinstance(self.broker.get_account().last_equity, float):
            #         last_equity = float(self.broker.get_account().last_equity)
            #     else:
            #         last_equity = self.broker.get_account().last_equity

            #     cur_time = time.time()
            #     self.equities.append([cur_time, last_equity])
            #     time.sleep(self.time_interval)

    def awaitMarketOpen(self):
        isOpen = self.broker.get_clock().is_open
        while not isOpen:
            clock = self.broker.get_clock()
            
            openingTime = clock.next_open.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()
            
            currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            timeToOpen = int((openingTime - currTime) / 60)
            
            self.logger.info(f"{str(timedelta( minutes = timeToOpen))} til market open.")
            time.sleep(60)
            isOpen = self.broker.get_clock().is_open

    def trade(self):
        state = self.get_state()

        if self.drl_lib == "elegantrl":
            with torch.no_grad():
                s_tensor = torch.as_tensor((state,), device=self.device)
                a_tensor = self.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]
            action = (action * self.max_stock).astype(int)

        elif self.drl_lib == "rllib":
            action = self.agent.compute_single_action(state)

        elif self.drl_lib == "stable_baselines3":
            action = self.model.predict(state)[0]

        else:
            raise ValueError(
                "The DRL library input is NOT supported yet. Please check your input."
            )

        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            min_action = 10  # stock_cd
            threads = []
            for index in np.where(action < -min_action)[0]:  # sell_index:
                sell_num_shares = min(self.stocks[index], -action[index])
                qty = abs(int(sell_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(
                        qty, self.stockUniverse[index], "sell", respSO
                    )
                )
                tSubmitOrder.start()
                threads.append(tSubmitOrder)  # record thread for joining later
                self.cash = float(self.broker.get_account().cash)
                self.stocks_cd[index] = 0

            for x in threads:  #  wait for all threads to complete
                x.join()

            threads = []
            for index in np.where(action > min_action)[0]:  # buy_index:
                if self.cash < 0:
                    tmp_cash = 0
                else:
                    tmp_cash = self.cash
                buy_num_shares = min(
                    tmp_cash // self.price[index], abs(int(action[index]))
                )
                if buy_num_shares != buy_num_shares:  # if buy_num_change = nan
                    qty = 0  # set to 0 quantity
                else:
                    qty = abs(int(buy_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(
                        qty, self.stockUniverse[index], "buy", respSO
                    )
                )
                tSubmitOrder.start()
                threads.append(tSubmitOrder)  # record thread for joining later
                self.cash = float(self.broker.get_account().cash)
                self.stocks_cd[index] = 0

            for x in threads:  #  wait for all threads to complete
                x.join()

        else:  # sell all when turbulence
            threads = []
            positions = self.broker.list_positions()
            for position in positions:
                if position.side == "long":
                    orderSide = "sell"
                else:
                    orderSide = "buy"
                qty = abs(int(float(position.qty)))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(qty, position.symbol, orderSide, respSO)
                )
                tSubmitOrder.start()
                threads.append(tSubmitOrder)  # record thread for joining later

            for x in threads:  #  wait for all threads to complete
                x.join()

            self.stocks_cd[:] = 0

    def get_state(self):
        self.logger.info ( 'get state')
        price, tech, turbulence = self.broker.fetch_latest_data(
            ticker_list=self.stockUniverse,
            time_interval=self.time_interval,  # "1Min", "5Min", "15Min", "1H", "1D
            tech_indicator_list=self.tech_indicator_list,
        )
        turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0

        turbulence = (
            self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2**-5
        ).astype(np.float32)

        tech = tech * 2**-7
        positions = self.broker.list_positions()
        stocks = [0] * len(self.stockUniverse)
        
        # filter positions in stockUniverse
        def filter_positions(position):
            return position.symbol in self.stockUniverse
        filtered_positions = list(filter(filter_positions, positions))

        for position in filtered_positions:
            ind = self.stockUniverse.index(position.symbol)
            stocks[ind] = abs(int(float(position.qty)))

        stocks = np.asarray(stocks, dtype=float)
        cash = float(self.broker.get_account().cash)
        self.cash = cash
        self.stocks = stocks
        self.turbulence_bool = turbulence_bool
        self.price = price

        amount = np.array(self.cash * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        state = np.hstack(
            (
                amount,
                turbulence,
                self.turbulence_bool,
                price * scale,
                self.stocks * scale,
                self.stocks_cd,
                tech,
            )
        ).astype(np.float32)
        state[np.isnan(state)] = 0.0
        state[np.isinf(state)] = 0.0
        return state

    def submitOrder(self, qty, stock, side, resp):
        if qty > 0:
            try:
                self.broker.submit_order(stock, qty, side, "market", "day")
                # self.broker.submit_order(stock, qty, side, "market")
                self.logger.info ( f"Market order of | {qty} {stock} {side} | completed.")
                resp.append(True)
            except Exception as e:
                self.logger.error ( f"Order of | {qty} {stock} {side} | did not go through.")
                self.logger.error ( e)
                resp.append(False)
        else:
            """
            self.logger.info(
                "Quantity is 0, order of | "
                + str(qty)
                + " "
                + stock
                + " "
                + side
                + " | not completed."
            )
            """
            resp.append(True)

    def close_conn(self):
        self.broker.close_conn()

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh


    

class StockEnvEmpty(gym.Env):
    # Empty Env used for loading rllib agent
    def __init__(self, config):
        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        self.env_num = 1
        self.max_step = 10000
        self.env_name = "StockEnvEmpty"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_discrete = False
        self.target_return = 9999
        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        return

    def step(self, actions):
        return