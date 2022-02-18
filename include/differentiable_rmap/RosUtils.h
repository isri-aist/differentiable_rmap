/* Author: Masaki Murooka */

/** \file RosUtils.h
    ROS utilities.
 */

#pragma once

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <differentiable_rmap/SamplingUtils.h>

namespace DiffRmap
{
/*! \brief Load message from rosbag.
 *  \tparam MsgType message type
 *  \param bag_path path of bag file
 */
template<class MsgType>
inline typename MsgType::ConstPtr loadBag(const std::string & bag_path)
{
  // find message
  rosbag::Bag bag(bag_path, rosbag::bagmode::Read);
  typename MsgType::ConstPtr msg_ptr = nullptr;
  int msg_count = 0;
  for(const auto & msg : rosbag::View(bag))
  {
    if(msg.isType<MsgType>())
    {
      msg_ptr = msg.instantiate<MsgType>();
      msg_count++;
    }
  }

  // check if message is loaded
  if(msg_count == 0)
  {
    mc_rtc::log::error_and_throw<std::runtime_error>("[loadBag] Failed to load sample set message from rosbag.");
  }
  else if(msg_count > 1)
  {
    ROS_WARN("[loadBag] Multiple messages are stored in bag file. load only last one.");
  }

  return msg_ptr;
}

/** \brief Variable manager based on ROS subscription
    \tparam MsgType ROS message type
    \tparam ValueType value type of managed variable
*/
template<class MsgType, class ValueType>
class SubscVariableManager
{
public:
  /** \brief Constructor.
      \param topic_name topic name
   */
  SubscVariableManager(const std::string & topic_name) : topic_name_(topic_name)
  {
    sub_ = nh_.subscribe(topic_name_, 1, &SubscVariableManager::callback, this);
  }

  /** \brief Constructor.
      \param topic_name topic name
      \param initial_value initial value
   */
  SubscVariableManager(const std::string & topic_name, const ValueType & initial_value)
  : SubscVariableManager(topic_name)
  {
    value_ = initial_value;
  }

  /** \brief Get value. */
  ValueType value() const
  {
    return value_;
  }

  /** \brief Set value. */
  void setValue(const ValueType & value)
  {
    value_ = value;
    has_new_value_ = true;
  }

  /** \brief Get whether the manager has new value. */
  bool hasNewValue() const
  {
    return has_new_value_;
  }

  /** \brief Make the flag of new value false. */
  void update()
  {
    has_new_value_ = false;
  }

protected:
  /** \brief ROS callback. */
  void callback(const typename MsgType::ConstPtr & msg)
  {
    setValue(msg->data);
  }

public:
  //! Topic name
  std::string topic_name_;

  //! Value
  ValueType value_;

  //! Whether the manager has new value
  bool has_new_value_ = false;

  //! ROS related members
  ros::NodeHandle nh_;

  ros::Subscriber sub_;
};
} // namespace DiffRmap
