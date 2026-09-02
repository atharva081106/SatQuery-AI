import React from 'react';
import styles from './SystemLoader.module.css';

export default function SystemLoader() {
  return (
    <div className={styles.loaderContainer}>
      <div className={styles.hole}>
        <i></i>
        <i></i>
        <i></i>
        <i></i>
        <i></i>
        <i></i>
        <i></i>
        <i></i>
        <i></i>
        <i></i>
      </div>
      <div className={styles.text}>INITIATING HYPERDRIVE...</div>
    </div>
  );
}
