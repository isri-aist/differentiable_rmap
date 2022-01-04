/* Author: Masaki Murooka */

#pragma once

#include <differentiable_rmap/SamplingUtils.h>


namespace DiffRmap
{
/** \brief Variable manager based on ROS subscription
    \tparam MsgType ROS message type
    \tparam ValueType value type of managed variable
*/
template <class MsgType, class ValueType>
class SubscVariableManager
{
 public:
  /** \brief Constructor.
      \param topic_name topic name
   */
  SubscVariableManager(const std::string& topic_name):
      topic_name_(topic_name)
  {
    sub_ = nh_.subscribe(topic_name_, 1, &SubscVariableManager::callback, this);
  }

  /** \brief Constructor.
      \param topic_name topic name
      \param initial_value initial value
   */
  SubscVariableManager(const std::string& topic_name,
                       const ValueType& initial_value):
      SubscVariableManager(topic_name)
  {
    value_ = initial_value;
  }

  /** \brief Get value. */
  ValueType value() const
  {
    return value_;
  }

  /** \brief Set value. */
  void setValue(const ValueType& value)
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
  void callback(const typename MsgType::ConstPtr& msg)
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
}
